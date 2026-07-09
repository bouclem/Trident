# Metrics

Track project metrics over time. Compare to previous state, flag regressions.

| Metric         | Current | Previous | Notes |
|----------------|---------|----------|-------|
| Version        | 0.0.4   | 0.0.3    | Tensor views + typed indexing |
| LOC            | ~1350   | ~1200    | Views + indexing + tests |
| Source files   | 5       | 5        | No new files |
| Test count     | 57      | 52       | Added view and indexing coverage |
| Test speed     | 5 ms    | 5 ms     | +14 tests, negligible overhead |
| Test coverage  | ~0%     | —        | Only core/tensor tested; formal coverage tooling TODO |
| Dependencies   | 0       | —        | No external deps |
| Build time     | ~3 s    | ~3 s     | Debug build, MSVC 19.44 |

---

<!-- Update after each milestone -->
