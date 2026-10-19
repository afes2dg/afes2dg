# Bolt's Journal - Critical Learnings

## 2023-10-19 - 3-Way Quick Sort Partitioning & Duplicate Handling
**Learning:** Naive quicksort implementations that partition into `[x for x in arr if x < pivot]` and `[x for x in arr if x > pivot]` inadvertently drop duplicate elements equal to `pivot`, causing data loss and incorrect sorting. Partitioning into 3 buckets (`left`, `middle`, `right`) in a single pass preserves all elements, correctly handles duplicates, and avoids repeated array traversals.
**Action:** Always use 3-way partitioning (`left`, `middle`, `right`) with a single loop pass when implementing quicksort in Python.
