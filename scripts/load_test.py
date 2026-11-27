"""Simulate several devices querying the live API at the same time.

Requires a running API server (``uvicorn app.main:app``), Postgres, and at
least one indexed book. Example:

    python scripts/load_test.py --concurrent 10

Each simulated device sends one simple question once. The script reports the
answer, cited sources, latency, and peak PostgreSQL connections so you can
verify that concurrent users each get their own information and that the DB
pool is not exhausted.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import psycopg
from sqlalchemy.engine import make_url

# Allow `python scripts/load_test.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is artificial intelligence?",
    "What is a neural network?",
    "What is deep learning?",
    "What is machine learning?",
    "What is natural language processing?",
    "What is reinforcement learning?",
    "What is computer vision?",
    "What is consciousness?",
    "What is a robot?",
    "What is the Turing test?",
]


@dataclass
class Activity:
    total: int = 0
    idle: int = 0
    in_transaction: int = 0
    active: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/api/query", help="query endpoint"
    )
    parser.add_argument(
        "--concurrent", type=int, default=10, help="simultaneous devices"
    )
    parser.add_argument(
        "--baseline", type=int, default=3, help="sequential requests to time first"
    )
    parser.add_argument("--top-k", type=int, default=3, help="sources per answer")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout")
    return parser.parse_args()


def db_connection():
    """Open a monitoring connection from the configured DATABASE_URL."""
    url = make_url(get_settings().database_url)
    return psycopg.connect(
        host=url.host,
        port=url.port or 5432,
        dbname=url.database,
        user=url.username,
        password=url.password,
        autocommit=True,
    )


def read_activity(conn, own_pid: int, database: str) -> Activity:
    with conn.cursor() as cur:
        cur.execute(
            "select pid, state from pg_stat_activity where datname = %s",
            (database,),
        )
        rows = [row for row in cur.fetchall() if row[0] != own_pid]
    return Activity(
        total=len(rows),
        idle=sum(state == "idle" for _, state in rows),
        in_transaction=sum(state == "idle in transaction" for _, state in rows),
        active=sum(state == "active" for _, state in rows),
    )


def monitor(conn, stop: threading.Event, samples: list[Activity]) -> None:
    with conn.cursor() as cur:
        cur.execute("select pg_backend_pid()")
        own_pid = cur.fetchone()[0]
    database = make_url(get_settings().database_url).database
    while not stop.is_set():
        samples.append(read_activity(conn, own_pid, database))
        stop.wait(0.2)


def ask(url: str, question: str, top_k: int, timeout: float, barrier=None) -> dict:
    if barrier is not None:
        barrier.wait()  # all devices fire at the same moment
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url, json={"question": question, "top_k": top_k}
            )
        elapsed = time.perf_counter() - started
        body = response.json() if response.status_code == 200 else {}
        return {
            "status": response.status_code,
            "elapsed": elapsed,
            "question": question,
            "answer": body.get("answer", ""),
            "source_ids": tuple(s["chunk_id"] for s in body.get("sources", [])),
            "detail": "" if response.status_code == 200 else response.text[:200],
        }
    except Exception as exc:  # noqa: BLE001 - report any client-side failure
        return {
            "status": None,
            "elapsed": time.perf_counter() - started,
            "question": question,
            "answer": "",
            "source_ids": (),
            "detail": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    if args.concurrent < 1 or args.baseline < 0 or args.top_k < 1:
        raise SystemExit("--concurrent and --top-k must be >= 1; --baseline >= 0")

    questions = [
        DEFAULT_QUESTIONS[i % len(DEFAULT_QUESTIONS)]
        for i in range(args.concurrent)
    ]

    baseline_times: list[float] = []
    for question in questions[: args.baseline]:
        result = ask(args.url, question, args.top_k, args.timeout)
        baseline_times.append(result["elapsed"])
        print(f"baseline: {result['status']} | {result['elapsed']:5.2f}s | {question}")
    if baseline_times:
        print(f"baseline average: {sum(baseline_times) / len(baseline_times):.2f}s")

    stop = threading.Event()
    samples: list[Activity] = []
    conn = None
    watcher = None
    try:
        conn = db_connection()
        watcher = threading.Thread(
            target=monitor, args=(conn, stop, samples), daemon=True
        )
        watcher.start()
    except Exception as exc:  # noqa: BLE001 - DB monitoring is optional
        print(f"DB monitoring unavailable: {type(exc).__name__}: {exc}")

    barrier = threading.Barrier(args.concurrent)
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        results = list(
            pool.map(
                lambda q: ask(args.url, q, args.top_k, args.timeout, barrier),
                questions,
            )
        )

    stop.set()
    if watcher is not None:
        watcher.join(timeout=2)
    if conn is not None:
        conn.close()

    print("\n=== simultaneous devices ===")
    for device, result in enumerate(results, start=1):
        print(
            f"device {device:>2} | {result['status']} | {result['elapsed']:5.2f}s | "
            f"sources={len(result['source_ids'])} | {result['question']}"
        )
        if result["detail"]:
            print(f"    error: {result['detail']}")
        elif result["answer"]:
            print(f"    answer: {result['answer'][:200].replace(chr(10), ' ')}")

    ok = [r for r in results if r["status"] == 200]
    print("\n=== summary ===")
    print(f"successful: {len(ok)}/{args.concurrent}")
    if ok:
        print(
            f"latency: min={min(r['elapsed'] for r in ok):.2f}s "
            f"max={max(r['elapsed'] for r in ok):.2f}s "
            f"avg={sum(r['elapsed'] for r in ok) / len(ok):.2f}s"
        )
        print(f"distinct answers: {len(set(r['answer'] for r in ok))}/{len(ok)}")
        print(
            f"distinct source sets: "
            f"{len(set(r['source_ids'] for r in ok))}/{len(ok)}"
        )
    if samples:
        print(
            f"db connections: peak_total={max(s.total for s in samples)} "
            f"peak_in_tx={max(s.in_transaction for s in samples)} "
            f"peak_active={max(s.active for s in samples)}"
        )


if __name__ == "__main__":
    main()
