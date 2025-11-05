#include "gtest/gtest.h"
#include "tiered_cache/lru_scheduler.h"
#include "tiered_cache/cache_tier.h"
#include <vector>
#include <memory>

namespace mooncake {

// Test fixture specifically for the LruCacheScheduler.
class LruCacheSchedulerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("LruCacheSchedulerTest");
        FLAGS_logtostderr = 1;
        // 1. Create a topology of fake tiers for testing.
        // Tier 1: 100 bytes capacity, priority 0
        // Tier 2: 200 bytes capacity, priority 1
        // Tier 3: 300 bytes capacity, priority 2
        tiers_.push_back(TierView{1, MemoryType::DRAM, 100, 0, 0, {}});
        tiers_.push_back(TierView{2, MemoryType::DRAM, 200, 0, 1, {}});
        tiers_.push_back(TierView{3, MemoryType::DRAM, 300, 0, 2, {}});

        // 2. Initialize the scheduler with the fake topology.
        scheduler_ = std::make_shared<LruCacheScheduler>();
        scheduler_->InitTopology(tiers_);
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    std::vector<TierView> tiers_;
    std::shared_ptr<LruCacheScheduler> scheduler_;
};

// Test the initial placement logic of the scheduler.
TEST_F(LruCacheSchedulerTest, GetPlacement) {
    SchedulingContext ctx{};

    // 1. First placement should go to the highest priority tier (tier 1).
    auto placement1 = scheduler_->GetPlacement("key1", ctx);
    ASSERT_EQ(placement1, 1);
}

// Test basic eviction logic.
TEST_F(LruCacheSchedulerTest, EvictionLogic) {
    // 1. Put two keys into tier 1.
    scheduler_->OnPut("key1", 1, 1);
    scheduler_->OnPut("key2", 1, 1);

    // 2. Plan for eviction from tier 1.
    MovementPlan plan = scheduler_->EvictKey(1);

    // "key1" was put first, so it's the least recently used and should be evicted.
    ASSERT_EQ(plan.key_to_move, "key1");
    ASSERT_EQ(plan.from_tier_id, 1);
    // It should be moved to the next tier (tier 2).
    ASSERT_EQ(plan.to_tier_id, 2);
    LOG(INFO) << "Test basic eviction planning successfully.";
}

// Test that OnAccess correctly updates the LRU order.
TEST_F(LruCacheSchedulerTest, OnAccessAndUpdate) {
    // 1. Put two keys into tier 1.
    scheduler_->OnPut("key1", 1, 1);
    scheduler_->OnPut("key2", 1, 1);

    // 2. Access "key1", making it the most recently used.
    scheduler_->OnAccess("key1", 1);
    LOG(INFO) << "Accessed key1, updating its LRU status.";

    // 3. Plan for eviction again.
    MovementPlan plan = scheduler_->EvictKey(1);

    // Now, "key2" should be the least recently used and the candidate for eviction.
    ASSERT_EQ(plan.key_to_move, "key2");
    ASSERT_EQ(plan.from_tier_id, 1);
    ASSERT_EQ(plan.to_tier_id, 2);
    LOG(INFO) << "Test LRU order update on access successfully.";
}

// Test eviction planning when the target tier is also full.
TEST_F(LruCacheSchedulerTest, EvictionToFullTier) {
    // 1. Fill tier 1.
    scheduler_->OnPut("k1", 1, 1);
    scheduler_->OnPut("k2", 1, 1);

    // 2. Fill tier 2.
    scheduler_->OnPut("k3", 2, 1);

    // 3. Plan eviction from tier 1.
    MovementPlan plan = scheduler_->EvictKey(1);

    // The LRU item is "k1". It should be moved from tier 1 to tier 2,
    // even if tier 2 is full. The expectation is that the caller
    // will then trigger an eviction from tier 2.
    ASSERT_EQ(plan.key_to_move, "k1");
    ASSERT_EQ(plan.from_tier_id, 1);
    ASSERT_EQ(plan.to_tier_id, 2);

    // 4. Now, plan eviction from tier 2.
    MovementPlan plan2 = scheduler_->EvictKey(2);
    ASSERT_EQ(plan2.key_to_move, "k3");
    ASSERT_EQ(plan2.from_tier_id, 2);
    ASSERT_EQ(plan2.to_tier_id, 3);
    LOG(INFO) << "Test eviction to a full tier successfully.";
}

// Test the OnDelete method.
TEST_F(LruCacheSchedulerTest, OnDelete) {
    // 1. Put three keys into tier 1.
    scheduler_->OnPut("key1", 1, 1);
    scheduler_->OnPut("key2", 1, 1);
    scheduler_->OnPut("key3", 1, 1);

    // 2. Delete the middle key.
    scheduler_->OnDelete("key2", 1);

    // 3. Plan for eviction.
    MovementPlan plan = scheduler_->EvictKey(1);

    // "key1" is still the LRU item.
    ASSERT_EQ(plan.key_to_move, "key1");

    // 4. Simulate moving key1 to tier 2
    scheduler_->OnDelete("key1", 1);
    scheduler_->OnPut("key1", 2, 1);

    // 5. Plan for eviction again from tier 1. Now "key3" should be the candidate.
    MovementPlan plan2 = scheduler_->EvictKey(1);
    ASSERT_EQ(plan2.key_to_move, "key3");
    LOG(INFO) << "Test OnDelete successfully removes key from LRU list.";
}

// Test the promotion logic.
TEST_F(LruCacheSchedulerTest, PromoteKey) {
    // 1. Put three keys into tier 2.
    scheduler_->OnPut("key1", 2, 1);
    scheduler_->OnPut("key2", 2, 1);
    scheduler_->OnPut("key3", 2, 1);

    // 2. Access "key2" to make it the most recently used (hottest).
    scheduler_->OnAccess("key2", 2);
    LOG(INFO) << "Accessed key2 in tier 2, making it the hottest.";

    // 3. Plan for promotion into tier 1.
    MovementPlan plan = scheduler_->PromoteKey(1);

    // The hottest key in the next tier ("key2" in tier 2) should be promoted.
    ASSERT_EQ(plan.key_to_move, "key2");
    ASSERT_EQ(plan.from_tier_id, 2);
    ASSERT_EQ(plan.to_tier_id, 1);
    LOG(INFO) << "Test promotion planning successfully.";
}

// Test the eviction of bytes.
TEST_F(LruCacheSchedulerTest, EvictBytesTest) {
    LruCacheScheduler scheduler;
    std::vector<TierView> tiers = {
        TierView{1, MemoryType::DRAM, 1000, 0, 0, {}},
        TierView{2, MemoryType::DRAM, 1000, 0, 1, {}}
    };
    scheduler.InitTopology(tiers);

    // 1. Put key1, key2, key3, key4
    scheduler.OnPut("key1", 1, 100);
    scheduler.OnPut("key2", 1, 200);
    scheduler.OnPut("key3", 1, 150);
    scheduler.OnPut("key4", 1, 50);

    // 2. Release 300 bytes, need to evict key1 and key2
    auto plans1 = scheduler.EvictBytes(1, 300);
    ASSERT_EQ(plans1.size(), 2);

    EXPECT_EQ(plans1[0].key_to_move, "key1");
    EXPECT_EQ(plans1[0].from_tier_id, 1);
    EXPECT_EQ(plans1[0].to_tier_id, 2);

    EXPECT_EQ(plans1[1].key_to_move, "key2");
    EXPECT_EQ(plans1[1].from_tier_id, 1);
    EXPECT_EQ(plans1[1].to_tier_id, 2);

    // 3. Access key2, LRU: key1, key3, key4, key2
    scheduler.OnAccess("key2", 1);

    // 4. Release 300 bytes, need to evict key1 and key3
    auto plans2 = scheduler.EvictBytes(1, 250);
    ASSERT_EQ(plans2.size(), 2);
    EXPECT_EQ(plans2[0].key_to_move, "key1");
    EXPECT_EQ(plans2[1].key_to_move, "key3");

    // 5. Set topology with no space in tier 2
    std::vector<TierView> tiers_no_space = {
        TierView{1, MemoryType::DRAM, 1000, 0, 0, {}},
        TierView{2, MemoryType::DRAM, 1000, 950, 1, {}}
    };
    scheduler.UpdateTopology(tiers_no_space);

    // 6. key1 (100 bytes) can not put in tier 2, so key1 should be discarded
    auto plans3 = scheduler.EvictBytes(1, 100);
    ASSERT_EQ(plans3.size(), 1);
    EXPECT_EQ(plans3[0].key_to_move, "key1");
    EXPECT_EQ(plans3[0].to_tier_id, MovementPlan::DISCARD);

    // 7. Release 0 bytes, no eviction
    auto plans4 = scheduler.EvictBytes(1, 0);
    EXPECT_TRUE(plans4.empty());
}

}  // namespace mooncake

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}