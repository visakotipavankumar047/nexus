from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

# ponytail: in-process, resets on restart; export to Prometheus/OTel when running multiple workers


@dataclass
class _Stat:
    count: int = 0
    errors: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


_stats: defaultdict[str, _Stat] = defaultdict(_Stat)
_lock = Lock()


def record(route: str, status: int, latency_ms: float) -> None:
    with _lock:
        st = _stats[route]
        st.count += 1
        st.errors += status >= 500
        st.total_ms += latency_ms
        st.max_ms = max(st.max_ms, latency_ms)


def snapshot() -> dict[str, dict[str, float]]:
    with _lock:
        return {r: {"count": s.count, "errors": s.errors, "avg_ms": round(s.total_ms / s.count, 2),
                    "max_ms": round(s.max_ms, 2)} for r, s in _stats.items()}
