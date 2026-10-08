// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transfer_metadata_plugin.h"

#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <json/value.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

#include <random>

#ifdef USE_REDIS
#include <hiredis/hiredis.h>

#include <mutex>
#endif

#ifdef USE_HTTP
#include <curl/curl.h>
#endif

#ifdef USE_ETCD
#ifdef USE_ETCD_LEGACY
#include <etcd/SyncClient.hpp>
#else
#include <libetcd_wrapper.h>
#endif
#endif  // USE_ETCD

#include <cassert>
#include <condition_variable>
#include <deque>
#include <set>
#include <unordered_map>

#include "common.h"
#include "config.h"
#include "error.h"

// Helper function to parse JSON string using thread-safe CharReaderBuilder
static bool parseJsonString(const std::string &json_str, Json::Value &value,
                            std::string *error_msg = nullptr) {
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errs;

    bool success = reader->parse(
        json_str.data(), json_str.data() + json_str.size(), &value, &errs);
    if (!success && error_msg) {
        *error_msg = errs;
    }
    return success;
}

namespace mooncake {
#ifdef USE_REDIS
struct RedisStoragePlugin : public MetadataStoragePlugin {
    RedisStoragePlugin(const std::string &metadata_uri)
        : client_(nullptr), metadata_uri_(metadata_uri) {
        auto hostname_port = parseHostNameWithPort(metadata_uri);
        client_ =
            redisConnect(hostname_port.first.c_str(), hostname_port.second);
        if (!client_) {
            LOG(ERROR) << "RedisStoragePlugin: unable to connect "
                       << metadata_uri_;
            return;
        }
        if (client_->err) {
            LOG(ERROR) << "RedisStoragePlugin: unable to connect "
                       << metadata_uri_ << ": " << client_->errstr;
            redisFree(client_);
            client_ = nullptr;
            return;
        }
    }

    RedisStoragePlugin(const std::string &metadata_uri,
                       const std::string &username, const std::string &password,
                       const uint8_t &db_index)
        : RedisStoragePlugin(metadata_uri) {
        if (!client_) {
            return;
        }

        if (!password.empty()) {
            redisReply *reply = nullptr;
            if (!username.empty()) {
                reply = static_cast<redisReply *>(redisCommand(
                    client_, "AUTH %b %b", username.data(), username.size(),
                    password.data(), password.size()));
            } else {
                reply = static_cast<redisReply *>(redisCommand(
                    client_, "AUTH %b", password.data(), password.size()));
            }
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG(ERROR) << "RedisStoragePlugin: authentication failed for "
                           << metadata_uri_;
                freeReplyObject(reply);
                redisFree(client_);
                client_ = nullptr;
                return;
            }
            freeReplyObject(reply);
        }

        if (db_index != 0) {
            auto *reply = static_cast<redisReply *>(
                redisCommand(client_, "SELECT %d", db_index));
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG(ERROR) << "RedisStoragePlugin: failed to select database "
                           << (int)db_index << " for " << metadata_uri_;
                freeReplyObject(reply);
                redisFree(client_);
                client_ = nullptr;
                return;
            }
            freeReplyObject(reply);
        }
    }

    virtual ~RedisStoragePlugin() {
        if (client_) {
            redisFree(client_);
            client_ = nullptr;
        }
    }

    virtual bool get(const std::string &key, Json::Value &value) {
        std::lock_guard<std::mutex> lock(access_client_mutex_);
        if (!client_) return false;

        redisReply *resp =
            (redisReply *)redisCommand(client_, "GET %s", key.c_str());
        if (!resp) {
            LOG(ERROR) << "RedisStoragePlugin: unable to get " << key
                       << " from " << metadata_uri_;
            return false;
        }
        if (resp->type == REDIS_REPLY_ERROR) {
            LOG(ERROR) << "RedisStoragePlugin: get " << key << " rejected by "
                       << metadata_uri_ << ": "
                       << (resp->str ? resp->str : "unknown error");
            freeReplyObject(resp);
            return false;
        }
        if (!resp->str) {
            LOG(ERROR) << "RedisStoragePlugin: unable to get " << key
                       << " from " << metadata_uri_;
            freeReplyObject(resp);
            return false;
        }

        auto json_file = std::string(resp->str);
        freeReplyObject(resp);

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "RedisStoragePlugin: JSON parse error: " << errs;
            return false;
        }
        return true;
    }

    // Distinguishes a nil reply (key absent — kNotFound) from a connection
    // or command error (kUnavailable) so syncSegmentCache does not purge the
    // cache during a Redis outage.
    GetResult getWithStatus(const std::string &key,
                            Json::Value &value) override {
        std::lock_guard<std::mutex> lock(access_client_mutex_);
        if (!client_) return GetResult::kUnavailable;

        redisReply *resp =
            (redisReply *)redisCommand(client_, "GET %s", key.c_str());
        if (!resp) {
            LOG(ERROR) << "RedisStoragePlugin: unable to get " << key
                       << " from " << metadata_uri_;
            return GetResult::kUnavailable;
        }
        if (resp->type == REDIS_REPLY_ERROR) {
            LOG(ERROR) << "RedisStoragePlugin: get " << key << " rejected by "
                       << metadata_uri_ << ": "
                       << (resp->str ? resp->str : "unknown error");
            freeReplyObject(resp);
            return GetResult::kUnavailable;
        }
        if (!resp->str) {
            freeReplyObject(resp);
            return GetResult::kNotFound;
        }

        auto json_file = std::string(resp->str);
        freeReplyObject(resp);

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "RedisStoragePlugin: JSON parse error: " << errs;
            return GetResult::kUnavailable;
        }
        return GetResult::kFound;
    }

    virtual bool set(const std::string &key, const Json::Value &value) {
        std::lock_guard<std::mutex> lock(access_client_mutex_);
        if (!client_) return false;

        Json::FastWriter writer;
        const std::string json_file = writer.write(value);
        redisReply *resp = (redisReply *)redisCommand(
            client_, "SET %s %s", key.c_str(), json_file.c_str());
        if (!resp) {
            LOG(ERROR) << "RedisStoragePlugin: unable to put " << key
                       << " from " << metadata_uri_;
            return false;
        }
        if (resp->type == REDIS_REPLY_ERROR) {
            // The server received and rejected the write (READONLY after a
            // failover, OOM at maxmemory, MISCONF, NOAUTH). Reporting success
            // loses the write silently; the constructor already treats the
            // same reply type as failure for AUTH/SELECT.
            LOG(ERROR) << "RedisStoragePlugin: put " << key << " rejected by "
                       << metadata_uri_ << ": "
                       << (resp->str ? resp->str : "unknown error");
            freeReplyObject(resp);
            return false;
        }
        freeReplyObject(resp);
        return true;
    }

    virtual bool remove(const std::string &key) {
        std::lock_guard<std::mutex> lock(access_client_mutex_);
        if (!client_) return false;

        redisReply *resp =
            (redisReply *)redisCommand(client_, "DEL %s", key.c_str());
        if (!resp) {
            LOG(ERROR) << "RedisStoragePlugin: unable to remove " << key
                       << " from " << metadata_uri_;
            return false;
        }
        if (resp->type == REDIS_REPLY_ERROR) {
            LOG(ERROR) << "RedisStoragePlugin: remove " << key
                       << " rejected by " << metadata_uri_ << ": "
                       << (resp->str ? resp->str : "unknown error");
            freeReplyObject(resp);
            return false;
        }
        freeReplyObject(resp);
        return true;
    }

    redisContext *client_;
    const std::string metadata_uri_;
    std::mutex access_client_mutex_;
};
#endif  // USE_REDIS

#ifdef USE_HTTP
struct HTTPStoragePlugin : public MetadataStoragePlugin {
    explicit HTTPStoragePlugin(const std::string &metadata_uri)
        : metadata_uri_(metadata_uri) {
        global_init_once();
    }

    ~HTTPStoragePlugin() override = default;

    static void global_init_once() {
        static std::once_flag once;
        std::call_once(once, [] { curl_global_init(CURL_GLOBAL_ALL); });
    }

    struct ThreadLocalCurl {
        CURL *h = nullptr;
        ThreadLocalCurl() {
            h = curl_easy_init();
            if (!h)
                throw std::runtime_error(
                    "HTTPStoragePlugin: curl_easy_init failed; cannot "
                    "initialize HTTP storage plugin functionality.");
            curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(h, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
        }
        ~ThreadLocalCurl() {
            if (h) curl_easy_cleanup(h);
        }
        ThreadLocalCurl(const ThreadLocalCurl &) = delete;
        ThreadLocalCurl &operator=(const ThreadLocalCurl &) = delete;
    };

    static CURL *tl_easy() {
        thread_local ThreadLocalCurl tls;
        return tls.h;
    }

    static size_t writeCallback(void *contents, size_t size, size_t nmemb,
                                void *userp) {
        auto *out = static_cast<std::string *>(userp);
        out->append(static_cast<const char *>(contents), size * nmemb);
        return size * nmemb;
    }

    std::string encodeUrl(const std::string &key) const {
        CURL *h = tl_easy();
        char *esc =
            curl_easy_escape(h, key.c_str(), static_cast<int>(key.size()));
        std::string url = metadata_uri_ + "?key=" + (esc ? esc : "");
        if (esc) curl_free(esc);
        return url;
    }

    static inline bool is_200(long code) { return code == 200; }

    bool get(const std::string &key, Json::Value &value) override {
        CURL *h = tl_easy();
        curl_easy_reset(h);

        std::string readBody;
        char errbuf[CURL_ERROR_SIZE] = {0};

        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, 3000L);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 1500L);

        const std::string url = encodeUrl(key);
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &readBody);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);

        CURLcode rc = curl_easy_perform(h);
        if (rc != CURLE_OK) {
            LOG(ERROR) << "GET " << url << " curl: " << curl_easy_strerror(rc)
                       << " err: " << errbuf;
            return false;
        }

        long code = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
        if (!is_200(code)) {
            LOG(ERROR) << "GET " << url << " http=" << code
                       << " body: " << readBody;
            return false;
        }

        Json::CharReaderBuilder b;
        std::string errs;
        std::unique_ptr<Json::CharReader> r(b.newCharReader());
        if (!r->parse(readBody.data(), readBody.data() + readBody.size(),
                      &value, &errs)) {
            LOG(ERROR) << "GET " << url << " json parse error: " << errs;
            return false;
        }
        return true;
    }

    // Distinguishes HTTP 404 (key absent — kNotFound) from curl/network
    // errors and server-side 5xx (kUnavailable) so syncSegmentCache does not
    // purge the cache during an etcd/master outage.
    GetResult getWithStatus(const std::string &key,
                            Json::Value &value) override {
        CURL *h = tl_easy();
        curl_easy_reset(h);

        std::string readBody;
        char errbuf[CURL_ERROR_SIZE] = {0};

        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, 3000L);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 1500L);

        const std::string url = encodeUrl(key);
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &readBody);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);

        CURLcode rc = curl_easy_perform(h);
        if (rc != CURLE_OK) {
            LOG(ERROR) << "GET " << url << " curl: " << curl_easy_strerror(rc)
                       << " err: " << errbuf;
            return GetResult::kUnavailable;
        }

        long code = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
        if (code == 404) return GetResult::kNotFound;
        if (!is_200(code)) {
            LOG(ERROR) << "GET " << url << " http=" << code
                       << " body: " << readBody;
            return GetResult::kUnavailable;
        }

        Json::CharReaderBuilder b;
        std::string errs;
        std::unique_ptr<Json::CharReader> r(b.newCharReader());
        if (!r->parse(readBody.data(), readBody.data() + readBody.size(),
                      &value, &errs)) {
            LOG(ERROR) << "GET " << url << " json parse error: " << errs;
            return GetResult::kUnavailable;
        }
        return GetResult::kFound;
    }

    bool set(const std::string &key, const Json::Value &value) override {
        CURL *h = tl_easy();
        curl_easy_reset(h);

        Json::StreamWriterBuilder wb;
        wb["indentation"] = "";
        const std::string payload = Json::writeString(wb, value);

        std::string readBody;
        char errbuf[CURL_ERROR_SIZE] = {0};

        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, 3000L);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 1500L);

        const std::string url = encodeUrl(key);
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(h, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, payload.size());

        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &readBody);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);

        CURLcode rc = curl_easy_perform(h);
        curl_slist_free_all(headers);

        if (rc != CURLE_OK) {
            LOG(ERROR) << "PUT " << url << " curl: " << curl_easy_strerror(rc)
                       << " err: " << errbuf;
            return false;
        }

        long code = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
        if (!is_200(code)) {
            LOG(ERROR) << "PUT " << url << " http=" << code
                       << " body: " << readBody;
            return false;
        }
        return true;
    }

    // ---- DELETE ----
    bool remove(const std::string &key) override {
        CURL *h = tl_easy();
        curl_easy_reset(h);

        std::string readBody;
        char errbuf[CURL_ERROR_SIZE] = {0};

        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, 3000L);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 1500L);

        const std::string url = encodeUrl(key);
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, "DELETE");
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &readBody);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);

        CURLcode rc = curl_easy_perform(h);
        if (rc != CURLE_OK) {
            LOG(ERROR) << "DELETE " << url
                       << " curl: " << curl_easy_strerror(rc)
                       << " err: " << errbuf;
            return false;
        }

        long code = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
        if (!is_200(code)) {
            LOG(ERROR) << "DELETE " << url << " http=" << code
                       << " body: " << readBody;
            return false;
        }
        return true;
    }

   private:
    const std::string metadata_uri_;
};

#endif  // USE_HTTP

#ifdef USE_ETCD
#ifdef USE_ETCD_LEGACY
struct EtcdStoragePlugin : public MetadataStoragePlugin {
    EtcdStoragePlugin(const std::string &metadata_uri)
        : client_(metadata_uri), metadata_uri_(metadata_uri) {}

    virtual ~EtcdStoragePlugin() {}

    virtual bool get(const std::string &key, Json::Value &value) {
        auto resp = client_.get(key);
        if (!resp.is_ok()) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to get " << key << " from "
                       << metadata_uri_ << ": " << resp.error_message();
            return false;
        }
        auto json_file = resp.value().as_string();

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "EtcdStoragePlugin: JSON parse error: " << errs;
            return false;
        }
        return true;
    }

    // etcd v3 answers a range request for a missing key with an OK response
    // carrying no value, so an empty payload is an authoritative absence;
    // everything else that fails (RPC error, timeout, malformed payload) is
    // reported as unavailability, never as absence.
    GetResult getWithStatus(const std::string &key,
                            Json::Value &value) override {
        auto resp = client_.get(key);
        if (!resp.is_ok()) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to get " << key << " from "
                       << metadata_uri_ << ": " << resp.error_message();
            return GetResult::kUnavailable;
        }
        auto json_file = resp.value().as_string();
        if (json_file.empty()) return GetResult::kNotFound;

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "EtcdStoragePlugin: JSON parse error: " << errs;
            return GetResult::kUnavailable;
        }
        return GetResult::kFound;
    }

    virtual bool set(const std::string &key, const Json::Value &value) {
        Json::FastWriter writer;
        const std::string json_file = writer.write(value);
        auto resp = client_.put(key, json_file);
        if (!resp.is_ok()) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to set " << key << " from "
                       << metadata_uri_ << ": " << resp.error_message();
            return false;
        }
        return true;
    }

    virtual bool remove(const std::string &key) {
        auto resp = client_.rm(key);
        if (!resp.is_ok()) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to delete " << key
                       << " from " << metadata_uri_ << ": "
                       << resp.error_message();
            return false;
        }
        return true;
    }

    etcd::SyncClient client_;
    const std::string metadata_uri_;
};
#else
struct EtcdStoragePlugin : public MetadataStoragePlugin {
    EtcdStoragePlugin(const std::string &metadata_uri)
        : metadata_uri_(metadata_uri) {
        auto ret = NewEtcdClient((char *)metadata_uri_.c_str(), &err_msg_);
        if (ret) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to connect "
                       << metadata_uri_ << ": " << err_msg_;
            // free the memory for storing error message
            free(err_msg_);
            err_msg_ = nullptr;
        }
    }

    virtual ~EtcdStoragePlugin() { EtcdCloseWrapper(); }

    virtual bool get(const std::string &key, Json::Value &value) {
        char *json_data = nullptr;
        auto ret = EtcdGetWrapper((char *)key.c_str(), &json_data, &err_msg_);
        if (ret) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to get " << key << " in "
                       << metadata_uri_ << ": " << err_msg_;
            // free the memory for storing error message
            free(err_msg_);
            err_msg_ = nullptr;
            return false;
        }
        if (!json_data) {
            return false;
        }
        auto json_file = std::string(json_data);
        // free the memory allocated by EtcdGetWrapper
        free(json_data);

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "EtcdStoragePlugin: JSON parse error: " << errs;
            return false;
        }
        return true;
    }

    // EtcdGetWrapper reports a missing key as ret == 0 with no payload, so
    // that combination is an authoritative absence; a non-zero return (RPC
    // error, timeout) is unavailability, never absence.
    GetResult getWithStatus(const std::string &key,
                            Json::Value &value) override {
        char *json_data = nullptr;
        auto ret = EtcdGetWrapper((char *)key.c_str(), &json_data, &err_msg_);
        if (ret) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to get " << key << " in "
                       << metadata_uri_ << ": " << err_msg_;
            // free the memory for storing error message
            free(err_msg_);
            err_msg_ = nullptr;
            return GetResult::kUnavailable;
        }
        if (!json_data) {
            return GetResult::kNotFound;
        }
        auto json_file = std::string(json_data);
        // free the memory allocated by EtcdGetWrapper
        free(json_data);

        std::string errs;
        if (!parseJsonString(json_file, value, &errs)) {
            LOG(ERROR) << "EtcdStoragePlugin: JSON parse error: " << errs;
            return GetResult::kUnavailable;
        }
        return GetResult::kFound;
    }

    virtual bool set(const std::string &key, const Json::Value &value) {
        Json::FastWriter writer;
        const std::string json_file = writer.write(value);
        auto ret = EtcdPutWrapper((char *)key.c_str(),
                                  (char *)json_file.c_str(), &err_msg_);
        if (ret) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to set " << key << " in "
                       << metadata_uri_ << ": " << err_msg_;
            // free the memory for storing error message
            free(err_msg_);
            err_msg_ = nullptr;
            return false;
        }
        return true;
    }

    virtual bool remove(const std::string &key) {
        auto ret = EtcdDeleteWrapper((char *)key.c_str(), &err_msg_);
        if (ret) {
            LOG(ERROR) << "EtcdStoragePlugin: unable to remove " << key
                       << " in " << metadata_uri_ << ": " << err_msg_;
            // free the memory for storing error message
            free(err_msg_);
            err_msg_ = nullptr;
            return false;
        }
        return true;
    }

    const std::string metadata_uri_;
    char *err_msg_;
};
#endif
#endif  // USE_ETCD

std::pair<std::string, std::string> parseConnectionString(
    const std::string &conn_string) {
    std::pair<std::string, std::string> result;
    std::string proto = "etcd";
    std::string domain;
    std::size_t pos = conn_string.find("://");

    if (pos != std::string::npos) {
        proto = conn_string.substr(0, pos);
        domain = conn_string.substr(pos + 3);
    } else {
        domain = conn_string;
    }

    result.first = proto;
    result.second = domain;
    return result;
}

std::shared_ptr<MetadataStoragePlugin> MetadataStoragePlugin::Create(
    const std::string &conn_string) {
    auto parsed_conn_string = parseConnectionString(conn_string);
#ifdef USE_ETCD
    if (parsed_conn_string.first == "etcd") {
        return std::make_shared<EtcdStoragePlugin>(parsed_conn_string.second);
    }
#endif  // USE_ETCD

#ifdef USE_REDIS
    if (parsed_conn_string.first == "redis") {
        const char *username = std::getenv("MC_REDIS_USERNAME");
        std::string username_str = username ? username : "";

        const char *password = std::getenv("MC_REDIS_PASSWORD");
        std::string password_str = password ? password : "";

        uint8_t db_index = 0;
        const char *db_index_str = std::getenv("MC_REDIS_DB_INDEX");
        if (db_index_str) {
            try {
                int index = std::stoi(db_index_str);
                if (index >= 0 && index <= 255) {
                    db_index = static_cast<uint8_t>(index);
                } else {
                    LOG(WARNING) << "Invalid Redis DB index: " << index
                                 << ", using default 0";
                }
            } catch (const std::exception &e) {
                LOG(WARNING)
                    << "Failed to parse MC_REDIS_DB_INDEX: " << e.what()
                    << ", using default 0";
            }
        }

        return std::make_shared<RedisStoragePlugin>(
            parsed_conn_string.second, username_str, password_str, db_index);
    }
#endif  // USE_REDIS

#ifdef USE_HTTP
    if (parsed_conn_string.first == "http" ||
        parsed_conn_string.first == "https") {
        return std::make_shared<HTTPStoragePlugin>(
            conn_string);  // including prefix
    }
#endif  // USE_HTTP

    LOG(FATAL) << "Unable to find metadata storage plugin "
               << parsed_conn_string.first
               << " with conn string: " << conn_string;
    return nullptr;
}

static inline const std::string getNetworkAddress(struct sockaddr *addr) {
    if (addr->sa_family == AF_INET) {
        struct sockaddr_in *sock_addr = (struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        if (inet_ntop(addr->sa_family, &(sock_addr->sin_addr), ip,
                      INET_ADDRSTRLEN) != NULL)
            return std::string(ip) + ":" +
                   std::to_string(ntohs(sock_addr->sin_port));
    } else if (addr->sa_family == AF_INET6) {
        struct sockaddr_in6 *sock_addr = (struct sockaddr_in6 *)addr;
        char ip[INET6_ADDRSTRLEN];
        if (inet_ntop(addr->sa_family, &(sock_addr->sin6_addr), ip,
                      INET6_ADDRSTRLEN) != NULL)
            return std::string(ip) + ":" +
                   std::to_string(ntohs(sock_addr->sin6_port));
    }
    PLOG(ERROR) << "Failed to parse socket address";
    return "";
}

struct PendingSocketConnect {
    int fd = -1;
    int flags = 0;
    bool connected = false;
    int64_t deadline_ms = 0;
    std::string remote_address;
};

ssize_t sendFullyNoSignal(int fd, const void *buffer, size_t length) {
    const auto *position = static_cast<const char *>(buffer);
    size_t remaining = length;
    while (remaining != 0) {
        const ssize_t written = send(fd, position, remaining, MSG_NOSIGNAL);
        if (written < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (written <= 0) return written;
        position += written;
        remaining -= written;
    }
    return static_cast<ssize_t>(length);
}

int sendCommandFrame(int fd, const std::string &request) {
    const uint64_t length = request.size() + sizeof(uint8_t);
    const uint8_t type =
        static_cast<uint8_t>(HandShakeRequestType::TransferCommand);
    if (sendFullyNoSignal(fd, &length, sizeof(length)) !=
            static_cast<ssize_t>(sizeof(length)) ||
        sendFullyNoSignal(fd, &type, sizeof(type)) !=
            static_cast<ssize_t>(sizeof(type)) ||
        sendFullyNoSignal(fd, request.data(), request.size()) !=
            static_cast<ssize_t>(request.size()))
        return ERR_SOCKET;
    return 0;
}

int beginSocketConnect(struct addrinfo *addr, const std::string &source_ip,
                       PendingSocketConnect &pending) {
    int on = 1;
    pending.fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (pending.fd == -1) {
        PLOG(ERROR) << "SocketHandShakePlugin: socket()";
        return ERR_SOCKET;
    }
    const auto fail = [&](int status) {
        close(pending.fd);
        pending.fd = -1;
        return status;
    };
    if (setsockopt(pending.fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on))) {
        PLOG(ERROR) << "SocketHandShakePlugin: setsockopt(TCP_NODELAY)";
        return fail(ERR_SOCKET);
    }
    if (setsockopt(pending.fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on))) {
        PLOG(ERROR) << "SocketHandShakePlugin: setsockopt(SO_REUSEADDR)";
        return fail(ERR_SOCKET);
    }

    if (!source_ip.empty()) {
        struct addrinfo hints {};
        struct addrinfo *sources = nullptr;
        hints.ai_family = addr->ai_family;
        hints.ai_socktype = addr->ai_socktype;
        if (getaddrinfo(source_ip.c_str(), "0", &hints, &sources))
            return fail(ERR_DNS);
        bool bound = false;
        for (auto *source = sources; source; source = source->ai_next) {
            if (bind(pending.fd, source->ai_addr, source->ai_addrlen) == 0) {
                bound = true;
                break;
            }
        }
        freeaddrinfo(sources);
        if (!bound) return fail(ERR_SOCKET);
    }

    struct timeval timeout;
    timeout.tv_sec = 60;
    timeout.tv_usec = 0;
    if (setsockopt(pending.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout))) {
        PLOG(ERROR) << "SocketHandShakePlugin: setsockopt(SO_RCVTIMEO)";
        return fail(ERR_SOCKET);
    }

    // SO_RCVTIMEO does not bound connect(). Start every connection in
    // non-blocking mode so callers can either wait here or overlap the TCP
    // handshake with command planning before finishSocketConnect().
    pending.flags = fcntl(pending.fd, F_GETFL, 0);
    if (pending.flags == -1 ||
        fcntl(pending.fd, F_SETFL, pending.flags | O_NONBLOCK) == -1) {
        PLOG(ERROR) << "SocketHandShakePlugin: fcntl(O_NONBLOCK)";
        return fail(ERR_SOCKET);
    }

    pending.remote_address = getNetworkAddress(addr->ai_addr);
    pending.deadline_ms = getCurrentTimeInMilli() +
                          globalConfig().handshake_connect_timeout * 1000;
    if (connect(pending.fd, addr->ai_addr, addr->ai_addrlen) == 0) {
        pending.connected = true;
        return 0;
    }
    if (errno != EINPROGRESS) {
        PLOG(ERROR) << "SocketHandShakePlugin: connect()"
                    << pending.remote_address;
        return fail(ERR_SOCKET);
    }
    return 0;
}

int finishSocketConnect(PendingSocketConnect &pending) {
    const auto fail = [&](int status) {
        close(pending.fd);
        pending.fd = -1;
        return status;
    };
    if (!pending.connected) {
        struct pollfd pfd;
        pfd.fd = pending.fd;
        pfd.events = POLLOUT;
        while (true) {
            const int64_t remaining_ms =
                pending.deadline_ms - getCurrentTimeInMilli();
            const int ret =
                remaining_ms <= 0 ? 0 : poll(&pfd, 1, (int)remaining_ms);
            if (ret > 0) break;
            if (ret == 0) {
                errno = ETIMEDOUT;
                PLOG(ERROR) << "SocketHandShakePlugin: connect() "
                            << pending.remote_address;
                return fail(ERR_SOCKET);
            }
            if (errno != EINTR) {
                PLOG(ERROR) << "SocketHandShakePlugin: poll()";
                return fail(ERR_SOCKET);
            }
        }

        int conn_err = 0;
        socklen_t err_len = sizeof(conn_err);
        if (getsockopt(pending.fd, SOL_SOCKET, SO_ERROR, &conn_err, &err_len)) {
            PLOG(ERROR) << "SocketHandShakePlugin: getsockopt(SO_ERROR)";
            return fail(ERR_SOCKET);
        }
        if (conn_err) {
            errno = conn_err;
            PLOG(ERROR) << "SocketHandShakePlugin: connect()"
                        << pending.remote_address;
            return fail(ERR_SOCKET);
        }
    }

    if (fcntl(pending.fd, F_SETFL, pending.flags) == -1) {
        PLOG(ERROR) << "SocketHandShakePlugin: fcntl(restore flags)";
        return fail(ERR_SOCKET);
    }
    pending.connected = true;
    return 0;
}

struct SocketHandShakePlugin : public HandShakePlugin {
    SocketHandShakePlugin() : listener_running_(false), listen_fd_(-1) {
        auto &config = globalConfig();
        listen_backlog_ = config.handshake_listen_backlog;
    }

    void closeListen() {
        if (listen_fd_ >= 0) {
            // LOG(INFO) << "SocketHandShakePlugin: closing listen socket";
            close(listen_fd_);
            listen_fd_ = -1;
        }
    }

    struct CommandJob {
        int fd;
        std::string peer_address;
        std::string request;
        std::chrono::steady_clock::time_point accepted;
        std::chrono::steady_clock::time_point received;
    };

    struct CachedCommandConnection {
        int fd = -1;
        std::chrono::steady_clock::time_point idle_since;
    };

    struct CommandConnectionCache {
        std::mutex mutex;
        std::unordered_map<std::string, CachedCommandConnection> connections;
        bool stopping = false;
    };

    struct SocketPreparedCommand final : PreparedHandshakeCommand {
        SocketPreparedCommand(std::shared_ptr<CommandConnectionCache> cache,
                              PendingSocketConnect pending,
                              std::chrono::steady_clock::time_point started,
                              std::string cache_key, bool reusable, bool reused)
            : cache(std::move(cache)),
              pending(std::move(pending)),
              started(started),
              cache_key(std::move(cache_key)),
              reusable(reusable),
              reused(reused) {}

        ~SocketPreparedCommand() override {
            if (pending.fd < 0) return;
            if (reusable && reused && cache) {
                SocketHandShakePlugin::cacheCommandConnection(cache, cache_key,
                                                              pending.fd);
            } else {
                close(pending.fd);
            }
            pending.fd = -1;
        }

        int send(const std::string &request, std::string &response) override {
            const auto send_started = std::chrono::steady_clock::now();
            int ret = finishSocketConnect(pending);
            if (ret) return ret;
            const auto connected = std::chrono::steady_clock::now();
            ret = sendCommandFrame(pending.fd, request);
            if (ret) {
                close(pending.fd);
                pending.fd = -1;
                return ret;
            }
            const auto written = std::chrono::steady_clock::now();
            auto [type, wire_response] = readString(pending.fd);
            const auto read = std::chrono::steady_clock::now();
            if (type != HandShakeRequestType::TransferCommand) {
                close(pending.fd);
                pending.fd = -1;
                return ERR_SOCKET;
            }
            response = std::move(wire_response);
            const int completed_fd = pending.fd;
            pending.fd = -1;
            if (reusable && cache) {
                SocketHandShakePlugin::cacheCommandConnection(cache, cache_key,
                                                              completed_fd);
            } else {
                close(completed_fd);
            }
            const auto milliseconds = [](auto duration) {
                return std::chrono::duration<double, std::milli>(duration)
                    .count();
            };
            VLOG(1) << "prepared transfer command socket profile bytes="
                    << request.size() << " prepare_lead_ms="
                    << milliseconds(send_started - started)
                    << " finish_ms=" << milliseconds(connected - send_started)
                    << " write_ms=" << milliseconds(written - connected)
                    << " read_ms=" << milliseconds(read - written)
                    << " send_ms=" << milliseconds(read - send_started)
                    << " reused=" << reused;
            return 0;
        }

        std::shared_ptr<CommandConnectionCache> cache;
        PendingSocketConnect pending;
        std::chrono::steady_clock::time_point started;
        std::string cache_key;
        bool reusable;
        bool reused;
    };

    static std::string commandConnectionKey(const std::string &ip_or_host_name,
                                            uint16_t rpc_port,
                                            const std::string &source_ip) {
        return ip_or_host_name + ":" + std::to_string(rpc_port) + "|" +
               source_ip;
    }

    int takeCommandConnection(const std::string &cache_key) {
        int fd = -1;
        {
            std::lock_guard<std::mutex> lock(command_connection_cache_->mutex);
            auto entry = command_connection_cache_->connections.find(cache_key);
            if (entry == command_connection_cache_->connections.end())
                return -1;
            const bool expired =
                std::chrono::steady_clock::now() - entry->second.idle_since >
                std::chrono::seconds(30);
            fd = entry->second.fd;
            command_connection_cache_->connections.erase(entry);
            if (expired) {
                close(fd);
                return -1;
            }
        }

        char byte = 0;
        const ssize_t peeked =
            recv(fd, &byte, sizeof(byte), MSG_PEEK | MSG_DONTWAIT);
        if (peeked < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return fd;
        }
        close(fd);
        return -1;
    }

    static void cacheCommandConnection(
        const std::shared_ptr<CommandConnectionCache> &cache,
        const std::string &cache_key, int fd) {
        if (fd < 0) return;
        int replaced = -1;
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            if (cache->stopping) {
                replaced = fd;
            } else {
                auto [entry, inserted] = cache->connections.emplace(
                    cache_key, CachedCommandConnection{
                                   fd, std::chrono::steady_clock::now()});
                if (!inserted) {
                    replaced = entry->second.fd;
                    entry->second = {fd, std::chrono::steady_clock::now()};
                }
            }
        }
        if (replaced >= 0) close(replaced);
    }

    void closeCommandConnections() {
        std::unordered_map<std::string, CachedCommandConnection> connections;
        {
            std::lock_guard<std::mutex> lock(command_connection_cache_->mutex);
            command_connection_cache_->stopping = true;
            connections.swap(command_connection_cache_->connections);
        }
        for (const auto &[key, connection] : connections) {
            (void)key;
            close(connection.fd);
        }
    }

    bool enqueueCommand(int fd, std::string peer_address, std::string request,
                        std::chrono::steady_clock::time_point accepted,
                        std::chrono::steady_clock::time_point received) {
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            if (!on_command_callback_ || command_stopping_ ||
                command_queue_.size() >= kCommandQueueDepth)
                return false;
            command_queue_.push_back(CommandJob{fd, std::move(peer_address),
                                                std::move(request), accepted,
                                                received});
        }
        command_cv_.notify_one();
        return true;
    }

    struct IdleCommandConnection {
        int fd = -1;
        std::string peer_address;
        std::string request_buffer;
    };

    void returnCommandConnection(int fd, std::string peer_address,
                                 std::string request_buffer = {}) {
        if (fd < 0) return;
        {
            std::lock_guard<std::mutex> lock(command_idle_mutex_);
            if (!command_reuse_running_) {
                close(fd);
                return;
            }
            pending_idle_commands_.push_back(
                {fd, std::move(peer_address), std::move(request_buffer)});
        }
        const uint8_t wake = 1;
        if (command_wakeup_pipe_[1] >= 0) {
            const ssize_t wake_result =
                write(command_wakeup_pipe_[1], &wake, sizeof(wake));
            (void)wake_result;
        }
    }

    void commandReuseWorker() {
        std::vector<IdleCommandConnection> idle;
        while (command_reuse_running_) {
            {
                std::lock_guard<std::mutex> lock(command_idle_mutex_);
                for (auto &connection : pending_idle_commands_)
                    idle.push_back(std::move(connection));
                pending_idle_commands_.clear();
            }

            std::vector<struct pollfd> poll_fds(idle.size() + 1);
            poll_fds[0] = {command_wakeup_pipe_[0], POLLIN, 0};
            for (size_t i = 0; i < idle.size(); ++i) {
                poll_fds[i + 1] = {
                    idle[i].fd, static_cast<short>(POLLIN | POLLERR | POLLHUP),
                    0};
            }

            const int ready = poll(poll_fds.data(), poll_fds.size(), -1);
            if (ready < 0) {
                if (errno == EINTR) continue;
                PLOG(ERROR) << "SocketHandShakePlugin: command poll()";
                break;
            }
            if (poll_fds[0].revents & POLLIN) {
                uint8_t wake[64];
                while (read(command_wakeup_pipe_[0], wake, sizeof(wake)) > 0) {
                }
            }
            if (!command_reuse_running_) break;

            for (size_t i = idle.size(); i != 0; --i) {
                const short events = poll_fds[i].revents;
                if (events == 0) continue;
                IdleCommandConnection connection = std::move(idle[i - 1]);
                idle[i - 1] = std::move(idle.back());
                idle.pop_back();

                if (!(events & POLLIN)) {
                    close(connection.fd);
                    continue;
                }
                char first_byte = 0;
                const ssize_t peeked =
                    recv(connection.fd, &first_byte, sizeof(first_byte),
                         MSG_PEEK | MSG_DONTWAIT);
                if (peeked <= 0) {
                    close(connection.fd);
                    continue;
                }
                const auto accepted = std::chrono::steady_clock::now();
                const auto type =
                    readString(connection.fd, connection.request_buffer);
                const auto received = std::chrono::steady_clock::now();
                if (type != HandShakeRequestType::TransferCommand ||
                    !enqueueCommand(connection.fd,
                                    std::move(connection.peer_address),
                                    std::move(connection.request_buffer),
                                    accepted, received)) {
                    close(connection.fd);
                }
            }
        }

        command_reuse_running_ = false;
        for (const auto &connection : idle) close(connection.fd);
        std::vector<IdleCommandConnection> pending;
        {
            std::lock_guard<std::mutex> lock(command_idle_mutex_);
            pending.swap(pending_idle_commands_);
        }
        for (const auto &connection : pending) close(connection.fd);
    }

    void startCommandReuse() {
        if (command_reuse_running_) return;
        if (pipe(command_wakeup_pipe_)) {
            PLOG(ERROR) << "SocketHandShakePlugin: command pipe()";
            command_wakeup_pipe_[0] = -1;
            command_wakeup_pipe_[1] = -1;
            return;
        }
        for (const int fd : command_wakeup_pipe_) {
            const int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        command_reuse_running_ = true;
        command_reuse_worker_ = std::thread([this] { commandReuseWorker(); });
    }

    void stopCommandReuse() {
        if (command_reuse_running_.exchange(false)) {
            const uint8_t wake = 1;
            if (command_wakeup_pipe_[1] >= 0) {
                const ssize_t wake_result =
                    write(command_wakeup_pipe_[1], &wake, sizeof(wake));
                (void)wake_result;
            }
        }
        if (command_reuse_worker_.joinable()) command_reuse_worker_.join();
        for (int &fd : command_wakeup_pipe_) {
            if (fd >= 0) close(fd);
            fd = -1;
        }
    }

    void commandWorker() {
        while (true) {
            CommandJob job;
            OnReceiveCommand callback;
            {
                std::unique_lock<std::mutex> lock(command_mutex_);
                command_cv_.wait(lock, [this] {
                    return command_stopping_ || !command_queue_.empty();
                });
                if (command_stopping_ && command_queue_.empty()) return;
                job = std::move(command_queue_.front());
                command_queue_.pop_front();
                callback = on_command_callback_;
            }

            const auto started = std::chrono::steady_clock::now();
            std::string response;
            try {
                if (callback) callback(job.peer_address, job.request, response);
            } catch (const std::exception &exception) {
                LOG(ERROR) << "SocketHandShakePlugin: command failed: "
                           << exception.what();
                response.clear();
            } catch (...) {
                LOG(ERROR) << "SocketHandShakePlugin: command failed";
                response.clear();
            }
            const auto executed = std::chrono::steady_clock::now();
            const size_t request_bytes = job.request.size();
            const int write_status = writeString(
                job.fd, HandShakeRequestType::TransferCommand, response);
            if (write_status) {
                LOG(ERROR) << "SocketHandShakePlugin: failed to send command "
                              "response";
            }
            const auto responded = std::chrono::steady_clock::now();
            if (write_status) {
                close(job.fd);
            } else {
                returnCommandConnection(job.fd, std::move(job.peer_address),
                                        std::move(job.request));
            }
            const auto milliseconds = [](auto duration) {
                return std::chrono::duration<double, std::milli>(duration)
                    .count();
            };
            VLOG(1) << "transfer command owner socket profile bytes="
                    << request_bytes << " receive_ms="
                    << milliseconds(job.received - job.accepted)
                    << " queue_ms=" << milliseconds(started - job.received)
                    << " execute_ms=" << milliseconds(executed - started)
                    << " respond_ms=" << milliseconds(responded - executed)
                    << " total_ms=" << milliseconds(responded - job.accepted);
        }
    }

    void stopCommandWorkers() {
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            command_stopping_ = true;
        }
        command_cv_.notify_all();
        for (auto &worker : command_workers_) worker.join();
        command_workers_.clear();
        while (!command_queue_.empty()) {
            close(command_queue_.front().fd);
            command_queue_.pop_front();
        }
    }

    virtual ~SocketHandShakePlugin() {
        if (listener_running_) {
            listener_running_ = false;
            listener_.join();
        }
        stopCommandReuse();
        stopCommandWorkers();
        closeCommandConnections();
        closeListen();
    }

    virtual void registerOnConnectionCallBack(OnReceiveCallBack callback) {
        on_connection_callback_ = callback;
    }

    virtual void registerOnMetadataCallBack(OnReceiveCallBack callback) {
        on_metadata_callback_ = callback;
    }

    virtual void registerOnNotifyCallBack(OnReceiveCallBack callback) {
        on_notify_callback_ = callback;
    }

    virtual void registerOnProbeCallBack(OnReceiveCallBack callback) {
        on_probe_callback_ = callback;
    }

    virtual void registerOnCommandCallBack(OnReceiveCommand callback) {
        bool start_reuse = false;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            on_command_callback_ = std::move(callback);
            if (on_command_callback_ && command_workers_.empty()) {
                command_stopping_ = false;
                for (size_t i = 0; i < kCommandWorkerCount; ++i)
                    command_workers_.emplace_back([this] { commandWorker(); });
                start_reuse = true;
            }
        }
        if (start_reuse) startCommandReuse();
    }

    virtual int startDaemon(uint16_t listen_port, int sockfd) {
        if (listener_running_) {
            // LOG(INFO) << "SocketHandShakePlugin: listener already running";
            return 0;
        }

        int on = 1;

        if (sockfd >= 0) {
            listen_fd_ = sockfd;
        } else {
            listen_fd_ = socket(globalConfig().use_ipv6 ? AF_INET6 : AF_INET,
                                SOCK_STREAM, 0);
            if (listen_fd_ < 0) {
                PLOG(ERROR) << "SocketHandShakePlugin: socket()";
                return ERR_SOCKET;
            }

            struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            if (setsockopt(listen_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                           sizeof(timeout))) {
                PLOG(ERROR) << "SocketHandShakePlugin: setsockopt(SO_RCVTIMEO)";
                closeListen();
                return ERR_SOCKET;
            }

            if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on,
                           sizeof(on))) {
                PLOG(ERROR)
                    << "SocketHandShakePlugin: setsockopt(SO_REUSEADDR)";
                closeListen();
                return ERR_SOCKET;
            }

            if (globalConfig().use_ipv6) {
                sockaddr_in6 bind_address;
                memset(&bind_address, 0, sizeof(sockaddr_in6));
                bind_address.sin6_family = AF_INET6;
                bind_address.sin6_port = htons(listen_port);
                bind_address.sin6_addr = IN6ADDR_ANY_INIT;

                if (bind(listen_fd_, (sockaddr *)&bind_address,
                         sizeof(sockaddr_in6)) < 0) {
                    PLOG(ERROR) << "SocketHandShakePlugin: bind (port "
                                << listen_port << ")";
                    closeListen();
                    return ERR_SOCKET;
                }
            } else {
                sockaddr_in bind_address;
                memset(&bind_address, 0, sizeof(sockaddr_in));
                bind_address.sin_family = AF_INET;
                bind_address.sin_port = htons(listen_port);
                bind_address.sin_addr.s_addr = INADDR_ANY;

                if (bind(listen_fd_, (sockaddr *)&bind_address,
                         sizeof(sockaddr_in)) < 0) {
                    PLOG(ERROR) << "SocketHandShakePlugin: bind (port "
                                << listen_port << ")";
                    closeListen();
                    return ERR_SOCKET;
                }
            }
        }

        if (listen(listen_fd_, listen_backlog_)) {
            PLOG(ERROR) << "SocketHandShakePlugin: listen()";
            closeListen();
            return ERR_SOCKET;
        }

        listener_running_ = true;
        listener_ = std::thread([this]() {
            while (listener_running_) {
                sockaddr_storage addr{};
                socklen_t addr_len = sizeof(addr);
                int conn_fd = accept(listen_fd_, (sockaddr *)&addr, &addr_len);
                if (conn_fd < 0) {
                    if (errno != EWOULDBLOCK && errno != EINTR)
                        PLOG(ERROR) << "SocketHandShakePlugin: accept()";
                    continue;
                }

                if (addr.ss_family != AF_INET && addr.ss_family != AF_INET6) {
                    LOG(ERROR) << "SocketHandShakePlugin: unsupported socket "
                                  "type, should be AF_INET or AF_INET6";
                    close(conn_fd);
                    continue;
                }

                int no_delay = 1;
                if (setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay,
                               sizeof(no_delay))) {
                    PLOG(ERROR)
                        << "SocketHandShakePlugin: setsockopt(TCP_NODELAY)";
                    close(conn_fd);
                    continue;
                }

                struct timeval timeout;
                timeout.tv_sec = 5;
                timeout.tv_usec = 0;
                if (setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                               sizeof(timeout))) {
                    PLOG(ERROR)
                        << "SocketHandShakePlugin: setsockopt(SO_RCVTIMEO)";
                    close(conn_fd);
                    continue;
                }

                auto peer_hostname =
                    getNetworkAddress((struct sockaddr *)&addr);

                Json::Value local, peer;

                const auto accepted = std::chrono::steady_clock::now();
                auto [type, json_str] = readString(conn_fd);
                const auto received = std::chrono::steady_clock::now();
                if (type == HandShakeRequestType::Invalid) {
                    close(conn_fd);
                    continue;
                }

                if (type == HandShakeRequestType::TransferCommand) {
                    timeout.tv_sec = 60;
                    if (setsockopt(conn_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                                   sizeof(timeout))) {
                        close(conn_fd);
                        continue;
                    }
                    if (!enqueueCommand(conn_fd, std::move(peer_hostname),
                                        std::move(json_str), accepted,
                                        received)) {
                        writeString(conn_fd, type, {});
                        close(conn_fd);
                    }
                    continue;
                }

                std::string errs;
                if (!parseJsonString(json_str, peer, &errs)) {
                    LOG(ERROR)
                        << "SocketHandShakePlugin: failed to receive "
                           "handshake message, "
                           "malformed json format: "
                        << errs << ", json string length: " << json_str.size()
                        << ", json string content: " << json_str;
                    close(conn_fd);
                    continue;
                }

                // old protocol equals Connection type
                if (type == HandShakeRequestType::Connection ||
                    type == HandShakeRequestType::OldProtocol) {
                    if (on_connection_callback_)
                        on_connection_callback_(peer, local);
                } else if (type == HandShakeRequestType::Metadata) {
                    if (on_metadata_callback_)
                        on_metadata_callback_(peer, local);
                } else if (type == HandShakeRequestType::Notify) {
                    if (on_notify_callback_) on_notify_callback_(peer, local);
                } else if (type == HandShakeRequestType::Probe) {
                    if (on_probe_callback_) on_probe_callback_(peer, local);
                } else {
                    LOG(ERROR) << "SocketHandShakePlugin: unexpected handshake "
                                  "message type";
                    close(conn_fd);
                    continue;
                }

                int ret =
                    writeString(conn_fd, type, Json::FastWriter{}.write(local));
                if (ret) {
                    LOG(ERROR) << "SocketHandShakePlugin: failed to send "
                                  "message: "
                                  "malformed json format, check tcp connection";
                    close(conn_fd);
                    continue;
                }

                ret = shutdown(conn_fd, SHUT_WR);
                if (ret) {
                    PLOG(ERROR) << "SocketHandShakePlugin: shutdown() failed, "
                                   "connection may be incomplete";
                    close(conn_fd);
                    continue;
                }

                // Wait for the client to close the connection
                char byte;
                ssize_t rc = read(conn_fd, &byte, sizeof(byte));
                if (rc > 0) {
                    LOG(ERROR) << "Unexpected socket read result: " << rc
                               << ", byte: " << int(byte);
                } else if (rc < 0) {
                    PLOG(ERROR)
                        << "Socket read failed while waiting client to close";
                }
                // else rc == 0, client close the connection, safe to close.

                close(conn_fd);
            }
            return;
        });

        return 0;
    }

    virtual int sendNotify(std::string ip_or_host_name, uint16_t rpc_port,
                           const Json::Value &local, Json::Value &peer) {
        struct addrinfo hints;
        struct addrinfo *result, *rp;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result)) {
            PLOG(ERROR)
                << "SocketHandShakePlugin: failed to get IP address of peer "
                   "server "
                << ip_or_host_name << ":" << rpc_port
                << ", check DNS and /etc/hosts, or use IPv4 address instead";
            return ERR_DNS;
        }

        int ret = 0;
        for (rp = result; rp; rp = rp->ai_next) {
            ret = doSendNotify(rp, local, peer);
            if (ret == 0) {
                freeaddrinfo(result);
                return 0;
            }
            if (ret == ERR_MALFORMED_JSON) {
                freeaddrinfo(result);
                return ret;
            }
        }

        freeaddrinfo(result);
        return ret;
    }

    virtual int sendProbe(std::string ip_or_host_name, uint16_t rpc_port,
                          const Json::Value &local, Json::Value &peer) {
        struct addrinfo hints;
        struct addrinfo *result, *rp;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result)) {
            PLOG(ERROR)
                << "SocketHandShakePlugin: failed to get IP address of peer "
                   "server "
                << ip_or_host_name << ":" << rpc_port
                << ", check DNS and /etc/hosts, or use IPv4 address instead";
            return ERR_DNS;
        }

        int ret = 0;
        for (rp = result; rp; rp = rp->ai_next) {
            ret = doSendProbe(rp, local, peer);
            if (ret == 0) {
                freeaddrinfo(result);
                return 0;
            }
            if (ret == ERR_MALFORMED_JSON) {
                freeaddrinfo(result);
                return ret;
            }
        }

        freeaddrinfo(result);
        return ret;
    }

    virtual int sendCommand(std::string ip_or_host_name, uint16_t rpc_port,
                            const std::string &source_ip,
                            const std::string &request, std::string &response) {
        struct addrinfo hints {};
        struct addrinfo *result = nullptr;
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result))
            return ERR_DNS;

        int ret = ERR_SOCKET;
        for (auto *address = result; address; address = address->ai_next) {
            ret = doSendCommand(address, source_ip, request, response);
            if (ret == 0) break;
        }
        freeaddrinfo(result);
        return ret;
    }

    std::unique_ptr<PreparedHandshakeCommand> prepareCommand(
        std::string ip_or_host_name, uint16_t rpc_port,
        const std::string &source_ip, bool reusable) override {
        const std::string cache_key =
            reusable
                ? commandConnectionKey(ip_or_host_name, rpc_port, source_ip)
                : std::string{};
        const auto started = std::chrono::steady_clock::now();
        if (reusable) {
            const int cached_fd = takeCommandConnection(cache_key);
            if (cached_fd >= 0) {
                PendingSocketConnect pending;
                pending.fd = cached_fd;
                pending.flags = fcntl(cached_fd, F_GETFL, 0);
                pending.connected = true;
                if (pending.flags >= 0) {
                    return std::make_unique<SocketPreparedCommand>(
                        command_connection_cache_, std::move(pending), started,
                        cache_key, true, true);
                }
                close(cached_fd);
            }
        }

        struct addrinfo hints {};
        struct addrinfo *result = nullptr;
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result))
            return {};

        std::unique_ptr<PreparedHandshakeCommand> prepared;
        for (auto *address = result; address; address = address->ai_next) {
            PendingSocketConnect pending;
            if (beginSocketConnect(address, source_ip, pending) == 0) {
                prepared = std::make_unique<SocketPreparedCommand>(
                    command_connection_cache_, std::move(pending), started,
                    cache_key, reusable, false);
                break;
            }
        }
        freeaddrinfo(result);
        return prepared;
    }

    virtual int send(std::string ip_or_host_name, uint16_t rpc_port,
                     const Json::Value &local, Json::Value &peer) {
        struct addrinfo hints;
        struct addrinfo *result, *rp;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result)) {
            PLOG(ERROR)
                << "SocketHandShakePlugin: failed to get IP address of peer "
                   "server "
                << ip_or_host_name << ":" << rpc_port
                << ", check DNS and /etc/hosts, or use IPv4 address instead";
            return ERR_DNS;
        }

        int ret = 0;
        for (rp = result; rp; rp = rp->ai_next) {
            ret = doSend(rp, local, peer);
            if (ret == 0) {
                freeaddrinfo(result);
                return 0;
            }
            if (ret == ERR_MALFORMED_JSON) {
                freeaddrinfo(result);
                return ret;
            }
        }

        freeaddrinfo(result);
        return ret;
    }

    int doConnect(struct addrinfo *addr, int &conn_fd,
                  const std::string &source_ip = {}) {
        PendingSocketConnect pending;
        int ret = beginSocketConnect(addr, source_ip, pending);
        if (ret) return ret;
        ret = finishSocketConnect(pending);
        if (ret) return ret;
        conn_fd = pending.fd;
        pending.fd = -1;
        return 0;
    }

    int doSend(struct addrinfo *addr, const Json::Value &local,
               Json::Value &peer) {
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd);
        if (ret) {
            return ret;
        }

        ret = writeString(conn_fd, HandShakeRequestType::Connection,
                          Json::FastWriter{}.write(local));
        if (ret) {
            LOG(ERROR)
                << "SocketHandShakePlugin: failed to send handshake message: "
                   "malformed json format, check tcp connection";
            close(conn_fd);
            return ret;
        }

        auto [type, json_str] = readString(conn_fd);
        if (type != HandShakeRequestType::Connection) {
            LOG(ERROR)
                << "SocketHandShakePlugin: unexpected handshake message type";
            close(conn_fd);
            return ERR_SOCKET;
        }

        std::string errs;
        if (!parseJsonString(json_str, peer, &errs)) {
            LOG(ERROR) << "SocketHandShakePlugin: failed to receive handshake "
                          "message: malformed json format: "
                       << errs;
            close(conn_fd);
            return ERR_MALFORMED_JSON;
        }

        close(conn_fd);
        return 0;
    }

    virtual int exchangeMetadata(std::string ip_or_host_name, uint16_t rpc_port,
                                 const Json::Value &local_metadata,
                                 Json::Value &peer_metadata) {
        struct addrinfo hints;
        struct addrinfo *result, *rp;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = globalConfig().use_ipv6 ? AF_INET6 : AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        char service[16];
        sprintf(service, "%u", rpc_port);
        if (getaddrinfo(ip_or_host_name.c_str(), service, &hints, &result)) {
            PLOG(ERROR)
                << "SocketHandShakePlugin: failed to get IP address of peer "
                   "server "
                << ip_or_host_name << ":" << rpc_port
                << ", check DNS and /etc/hosts, or use IPv4 address instead";
            return ERR_DNS;
        }

        int ret = 0;
        for (rp = result; rp; rp = rp->ai_next) {
            ret = doSendMetadata(rp, local_metadata, peer_metadata);
            if (ret == 0) {
                freeaddrinfo(result);
                return 0;
            }
            if (ret == ERR_MALFORMED_JSON) {
                freeaddrinfo(result);
                return ret;
            }
        }

        freeaddrinfo(result);
        return ret;
    }

    int doSendNotify(struct addrinfo *addr, const Json::Value &local_notify,
                     Json::Value &peer_notify) {
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd);
        if (ret) {
            return ret;
        }

        ret = writeString(conn_fd, HandShakeRequestType::Notify,
                          Json::FastWriter{}.write(local_notify));
        if (ret) {
            LOG(ERROR)
                << "SocketHandShakePlugin: failed to send metadata message: "
                   "malformed json format, check tcp connection";
            close(conn_fd);
            return ret;
        }

        auto [type, json_str] = readString(conn_fd);
        if (type != HandShakeRequestType::Notify) {
            LOG(ERROR)
                << "SocketHandShakePlugin: unexpected handshake message type";
            close(conn_fd);
            return ERR_SOCKET;
        }

        // LOG(INFO) << "SocketHandShakePlugin: received metadata message: "
        //           << json_str;

        std::string errs;
        if (!parseJsonString(json_str, peer_notify, &errs)) {
            LOG(ERROR) << "SocketHandShakePlugin: failed to receive metadata "
                          "message, malformed json format: "
                       << errs;
            close(conn_fd);
            return ERR_MALFORMED_JSON;
        }

        close(conn_fd);
        return 0;
    }

    int doSendProbe(struct addrinfo *addr, const Json::Value &local_probe,
                    Json::Value &peer_probe) {
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd);
        if (ret) {
            return ret;
        }

        ret = writeString(conn_fd, HandShakeRequestType::Probe,
                          Json::FastWriter{}.write(local_probe));
        if (ret) {
            LOG(ERROR)
                << "SocketHandShakePlugin: failed to send probe message: "
                   "malformed json format, check tcp connection";
            close(conn_fd);
            return ret;
        }

        auto [type, json_str] = readString(conn_fd);
        if (type != HandShakeRequestType::Probe) {
            LOG(ERROR)
                << "SocketHandShakePlugin: unexpected probe message type";
            close(conn_fd);
            return ERR_SOCKET;
        }

        std::string errs;
        if (!parseJsonString(json_str, peer_probe, &errs)) {
            LOG(ERROR) << "SocketHandShakePlugin: failed to receive probe "
                          "message, malformed json format: "
                       << errs;
            close(conn_fd);
            return ERR_MALFORMED_JSON;
        }

        close(conn_fd);
        return 0;
    }

    int doSendCommand(struct addrinfo *addr, const std::string &source_ip,
                      const std::string &request, std::string &response) {
        const auto started = std::chrono::steady_clock::now();
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd, source_ip);
        if (ret) return ret;
        const auto connected = std::chrono::steady_clock::now();

        ret = writeString(conn_fd, HandShakeRequestType::TransferCommand,
                          request);
        if (ret) {
            close(conn_fd);
            return ret;
        }
        const auto written = std::chrono::steady_clock::now();
        auto [type, wire_response] = readString(conn_fd);
        const auto read = std::chrono::steady_clock::now();
        if (type != HandShakeRequestType::TransferCommand) {
            close(conn_fd);
            return ERR_SOCKET;
        }
        response = std::move(wire_response);
        close(conn_fd);
        const auto milliseconds = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        VLOG(1) << "transfer command socket profile bytes=" << request.size()
                << " connect_ms=" << milliseconds(connected - started)
                << " write_ms=" << milliseconds(written - connected)
                << " read_ms=" << milliseconds(read - written)
                << " total_ms=" << milliseconds(read - started);
        return 0;
    }

    int doSendMetadata(struct addrinfo *addr, const Json::Value &local_metadata,
                       Json::Value &peer_metadata) {
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd);
        if (ret) {
            return ret;
        }

        ret = writeString(conn_fd, HandShakeRequestType::Metadata,
                          Json::FastWriter{}.write(local_metadata));
        if (ret) {
            LOG(ERROR)
                << "SocketHandShakePlugin: failed to send metadata message: "
                   "malformed json format, check tcp connection";
            close(conn_fd);
            return ret;
        }

        auto [type, json_str] = readString(conn_fd);
        if (type != HandShakeRequestType::Metadata) {
            LOG(ERROR)
                << "SocketHandShakePlugin: unexpected handshake message type";
            close(conn_fd);
            return ERR_SOCKET;
        }

        // LOG(INFO) << "SocketHandShakePlugin: received metadata message: "
        //           << json_str;

        std::string errs;
        if (!parseJsonString(json_str, peer_metadata, &errs)) {
            LOG(ERROR) << "SocketHandShakePlugin: failed to receive metadata "
                          "message, malformed json format: "
                       << errs;
            close(conn_fd);
            return ERR_MALFORMED_JSON;
        }

        close(conn_fd);
        return 0;
    }

    std::atomic<bool> listener_running_;
    std::thread listener_;
    int listen_fd_;
    int listen_backlog_;

    static constexpr size_t kCommandWorkerCount = 4;
    static constexpr size_t kCommandQueueDepth = 16;
    std::mutex command_mutex_;
    std::condition_variable command_cv_;
    std::deque<CommandJob> command_queue_;
    std::vector<std::thread> command_workers_;
    OnReceiveCommand on_command_callback_;
    bool command_stopping_ = false;

    std::mutex command_idle_mutex_;
    std::vector<IdleCommandConnection> pending_idle_commands_;
    std::atomic<bool> command_reuse_running_{false};
    std::thread command_reuse_worker_;
    int command_wakeup_pipe_[2] = {-1, -1};

    std::shared_ptr<CommandConnectionCache> command_connection_cache_ =
        std::make_shared<CommandConnectionCache>();

    OnReceiveCallBack on_connection_callback_;
    OnReceiveCallBack on_metadata_callback_;
    OnReceiveCallBack on_notify_callback_;
    OnReceiveCallBack on_probe_callback_;
};

std::shared_ptr<HandShakePlugin> HandShakePlugin::Create(
    const std::string &conn_string) {
    return std::make_shared<SocketHandShakePlugin>();
}

std::vector<std::string> findLocalIpAddresses() {
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        PLOG(ERROR) << "getifaddrs failed";
        return ips;
    }

    auto use_ipv6 = globalConfig().use_ipv6;
    sa_family_t family = use_ipv6 ? AF_INET6 : AF_INET;

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == family) {
            if (strcmp(ifa->ifa_name, "lo") == 0) {
                continue;
            }

            // Check if interface is UP and RUNNING
            if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) {
                LOG(INFO) << "Skipping interface " << ifa->ifa_name
                          << " (not UP or not RUNNING)";
                continue;
            }

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr,
                            use_ipv6 ? sizeof(struct sockaddr_in6)
                                     : sizeof(struct sockaddr_in),
                            host, NI_MAXHOST, nullptr, 0,
                            NI_NUMERICHOST) == 0) {
                LOG(INFO) << "Found active interface " << ifa->ifa_name
                          << " with IP " << host;
                ips.push_back(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

uint16_t findAvailableTcpPort(int &sockfd, bool set_range) {
    static std::random_device rand_gen;
    std::uniform_int_distribution rand_dist;
    int min_port = globalConfig().rpc_min_port;
    int max_port = globalConfig().rpc_max_port;
#ifdef USE_BAREX
    if (set_range) {
        min_port = 17000;
        max_port = 35000;
        const char *min_port_env = std::getenv("ACCL_MIN_PORT");
        const char *max_port_env = std::getenv("ACCL_MAX_PORT");
        if (min_port_env) {
            int val = atoi(min_port_env);
            if (val > 1024 && val < 65536) {
                min_port = val;
            }
        }
        if (max_port_env) {
            int val = atoi(max_port_env);
            if (val > 1024 && val < 65536 && val > min_port) {
                max_port = val;
            }
        }
    }
#endif
    const int max_attempts = 500;
    bool use_ipv6 = globalConfig().use_ipv6;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = min_port + rand_dist(rand_gen) % (max_port - min_port + 1);
        sockfd = socket(use_ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            continue;
        }

        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       sizeof(timeout))) {
            close(sockfd);
            sockfd = -1;
            continue;
        }

        if (use_ipv6) {
            sockaddr_in6 bind_address;
            memset(&bind_address, 0, sizeof(sockaddr_in6));
            bind_address.sin6_family = AF_INET6;
            bind_address.sin6_port = htons(port);
            bind_address.sin6_addr = IN6ADDR_ANY_INIT;
            if (bind(sockfd, (sockaddr *)&bind_address, sizeof(sockaddr_in6)) <
                0) {
                close(sockfd);
                sockfd = -1;
                continue;
            }
        } else {
            sockaddr_in bind_address;
            memset(&bind_address, 0, sizeof(sockaddr_in));
            bind_address.sin_family = AF_INET;
            bind_address.sin_port = htons(port);
            bind_address.sin_addr.s_addr = INADDR_ANY;
            if (bind(sockfd, (sockaddr *)&bind_address, sizeof(sockaddr_in)) <
                0) {
                close(sockfd);
                sockfd = -1;
                continue;
            }
        }

        return port;
    }
    return 0;
}

}  // namespace mooncake
