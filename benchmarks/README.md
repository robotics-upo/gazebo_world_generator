# Collision validation baseline

Run `PYTHONPATH=. python3 benchmarks/placement_baseline.py` from the repository root.
The fixture uses a fixed 10-column grid of 0.5 m objects separated by 0.3 m.
It exercises the collision validation path without an LLM or model lookup.

Host baseline (Python 3.10.12, 2026-09-23):

| Objects | Seconds | LLM calls | Overlaps |
| ---: | ---: | ---: | ---: |
| 10 | 0.000027 | 0 | 0 |
| 50 | 0.000333 | 0 | 0 |
| 100 | 0.001198 | 0 | 0 |

These times are a collision-check microbenchmark, not end-to-end placement
times. They do not justify changing the collision algorithm yet.

Run `PYTHONPATH=. python3 benchmarks/placement_pipeline.py` for the offline
warehouse placement pipeline. It uses a fixed local model catalog, seed 7,
three repetitions per size, and no LLM calls. Times are medians on the same
host; the overlap check uses the fixture model's 0.8 m by 0.4 m footprint.

| Objects | Before grid input fix (s) | After fix (s) | Placed | LLM calls | Overlaps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000800 | 0.000808 | 10 | 0 | 0 |
| 50 | 0.004979 | 0.005248 | 50 | 0 | 0 |
| 100 | 0.017505 | 0.018423 | 100 | 0 | 0 |

The repeated-object grid strategy previously failed because its inputs lacked
dimensions. The fix restores that strategy; it is a correctness repair and does
not improve this benchmark's timing. No collision optimization was accepted.
