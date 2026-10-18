## 2025-10-18 - Avoid synchronous gc.collect() in data processing functions
**Learning:** Synchronous `gc.collect()` inside hot numerical/tensor processing functions causes full garbage collection pauses ("stop-the-world" latency), degrading throughput by ~3x.
**Action:** Let Python's automatic garbage collector handle memory lifecycle, or trigger `gc.collect()` only in high-level background memory monitoring tasks when system RAM exceeds safe thresholds.
