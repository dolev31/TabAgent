"""Warm sandbox workers: the same results and isolation as SubprocessSandbox and the trusted in-process mode,
and the failure handling of a persistent pool (timeouts, crashes, restarts, recycling, shutdown)."""

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime

import pytest

from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.warm import WarmSandbox

POSIX = sys.platform != "win32"

# A representative compiled extractor: stdlib parsing of a chat prefix, as TabSchema writes them.
REALISTIC = textwrap.dedent(
    """
    import json, re
    from collections import Counter

    def extract_features_for_classification(trajectory_data):
        steps = trajectory_data['steps']
        users = [s['content'] for s in steps if s['name'] == 'user']
        calls = [c for s in steps for c in (s['tool_calls'] or [])]
        words = re.findall(r"[a-z]+", " ".join(users).lower())
        top = Counter(words).most_common(1)
        return {
            "goal": trajectory_data['instruction'],
            "n_user_turns": len(users),
            "n_tool_calls": len(calls),
            "last_tool": calls[-1] if calls else "",
            "top_word": top[0][0] if top else "",
            "mentions_flight": int("flight" in words),
            "payload_chars": len(json.dumps(trajectory_data, sort_keys=True)),
        }

    features = extract_features_for_classification(trajectory)
    """
)


def _prefix(i: int) -> dict:
    user = f"book a flight to city {i} please"
    steps = [{"name": "user", "content": user, "tool_calls": [], "tool_name": ""}]
    steps += [
        {"name": "assistant", "content": "", "tool_calls": [f"search_{j}"], "tool_name": ""}
        for j in range(i % 4)
    ]
    return {"instruction": f"goal {i}", "steps": steps, 1: "an int key", "when": datetime(2026, 1, i + 1)}


@pytest.fixture
def warm():
    box = WarmSandbox(size=2, timeout_s=20)
    yield box
    box.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a reused pid of another user
        return True
    return True


def _wait_until(cond, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


# ----------------------------------------------------------------------------------------------- parity
@pytest.mark.filterwarnings("ignore:the TabSchema extractor failed:RuntimeWarning")
def test_warm_cold_and_in_process_extraction_give_identical_features(warm):
    prefixes = [_prefix(i) for i in range(6)] + [{"steps": []}]  # the last one fails (no 'instruction')
    cold_res, warm_res, trusted_res = [], [], []
    cold = ExtractorRunner(REALISTIC, sandbox=SubprocessSandbox()).run(prefixes, results_out=cold_res)
    hot = ExtractorRunner(REALISTIC, sandbox=warm).run(prefixes, results_out=warm_res)
    trusted = ExtractorRunner(REALISTIC, trust_extractor=True).run(prefixes, results_out=trusted_res)
    assert hot == cold == trusted and hot[-1] is None and all(hot[:-1])
    assert [r.ok for r in warm_res] == [r.ok for r in cold_res] == [True] * 6 + [False]
    last = [r.error.strip().splitlines()[-1] for r in (warm_res[-1], cold_res[-1], trusted_res[-1])]
    assert last == ["KeyError: 'instruction'"] * 3
    assert warm_res[-1].kind == cold_res[-1].kind == trusted_res[-1].kind == "error"


def test_one_slow_trajectory_fails_alone():
    code = "if trajectory['slow']:\n    while True:\n        pass\nfeatures = {'x': trajectory['i']}\n"
    items = [{"slow": i == 3, "i": i} for i in range(7)]
    with WarmSandbox(size=2, timeout_s=1) as box:
        t0 = time.monotonic()
        res = box.run_batch(code, items)
        assert time.monotonic() - t0 < 15
    assert [r.ok for r in res] == [i != 3 for i in range(7)] and "timed out" in res[3].error
    assert [r.features for r in res if r.ok] == [{"x": i} for i in range(7) if i != 3]


def test_a_hang_that_ignores_the_alarm_fails_alone_and_the_worker_is_replaced():
    code = (
        "import signal\n"
        "if trajectory['hang']:\n"
        "    signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        "    while True:\n"
        "        pass\n"
        "features = {'x': trajectory['i']}\n"
    )
    with WarmSandbox(size=1, timeout_s=1) as box:
        res = [box.run(code, {"hang": i == 2, "i": i}) for i in range(4)]
        stats = box.stats()
    assert [r.ok for r in res] == [True, True, False, True]  # a cold chunk loses item 3 too
    assert "timed out" in res[2].error and "killed" in res[2].error
    assert stats["restarts"] == 1 and stats["timeouts"] == 1
    # the alarm the code disabled is set again for the next request
    with WarmSandbox(size=1, timeout_s=1) as box:
        box.run("import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nfeatures = {}", {})
        slow = box.run("while True:\n    pass\n", {})
        assert "TimeoutError" in slow.error and box.stats()["restarts"] == 0


def test_warm_workers_never_see_api_keys(monkeypatch, warm):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "t" * 30)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a" * 30)
    code = "import os\nfeatures = {'keys': ','.join(k for k in os.environ if 'KEY' in k or 'TOKEN' in k)}"
    r = warm.run(code, {})
    assert r.ok and r.features == {"keys": ""}


def test_exit_calls_fail_only_their_own_request(warm):
    code = (
        "import os\n"
        "mode = trajectory['mode']\n"
        "if mode == 'exit':\n    exit(3)\n"
        "if mode == 'raise':\n    raise SystemExit(4)\n"
        "if mode == 'interrupt':\n    raise KeyboardInterrupt\n"
        "if mode == 'os_exit':\n    os._exit(5)\n"
        "features = {'mode': mode}\n"
    )
    modes = ["ok", "exit", "raise", "interrupt", "ok", "os_exit", "ok"]
    res = [warm.run(code, {"mode": m}) for m in modes]
    assert [r.ok for r in res] == [True, False, False, False, True, False, True]
    assert "SystemExit" in res[2].error and "exited early (exit status 5)" in res[5].error
    assert warm.stats()["restarts"] == 1


def test_printing_or_reading_stdin_cannot_corrupt_the_protocol(warm):
    code = (
        "import os, sys\n"
        "print(trajectory)\n"
        'print(\'{"ok": true, "features": {"forged": 1}}\')\n'
        "print('{not json')\n"
        'os.write(1, b\'{"ok": false, "error": "forged", "id": 1}\\n\')\n'
        "sys.stdout.write('x' * 200000)\n"  # more than a pipe buffer: the parent drains it
        "rest = sys.stdin.read()\n"  # the next request is not on stdin
        "features = {'n': trajectory['n'], 'stdin': rest}\n"
        "print(features, '🙂')\n"
    )
    res = warm.run_batch(code, [{"n": i} for i in range(5)])
    assert [r.features for r in res] == [{"n": i, "stdin": ""} for i in range(5)]
    eof = warm.run("x = input()\nfeatures = {'x': x}", {})
    assert not eof.ok and "EOFError" in eof.error


def test_recursion_bomb_fails_cleanly(warm):
    r = warm.run("def f(n):\n    return f(n + 1)\nfeatures = {'x': f(0)}", {})
    assert not r.ok and "RecursionError" in r.error
    assert warm.run("features = {'ok': 1}", {}).features == {"ok": 1}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced on Linux only")
def test_memory_bomb_is_capped():
    with WarmSandbox(size=1, timeout_s=30, mem_mb=512) as box:
        r = box.run("x = bytearray(4 * 1024 ** 3)\nfeatures = {'n': len(x)}", {})
        assert not r.ok and "MemoryError" in r.error
        assert box.run("features = {'ok': 1}", {}).ok


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="macOS does not enforce RLIMIT_CPU")
def test_the_cpu_limit_is_per_request():
    burn = (
        "import time\nt = time.process_time()\nwhile time.process_time() - t < trajectory['s']:\n    pass\n"
    )
    with WarmSandbox(size=1, timeout_s=20, cpu_s=1) as box:
        # three requests use more CPU in total than cpu_s, each one less: none is stopped
        assert all(box.run(burn + "features = {}", {"s": 0.6}).ok for _ in range(3))
        hog = box.run(burn + "features = {}", {"s": 5})
        assert not hog.ok and "exited early" in hog.error  # SIGXCPU
        assert box.run("features = {'ok': 1}", {}).ok


def test_unserializable_features_and_compile_errors_are_failures(warm):
    assert "TypeError" in warm.run("features = [1]", {}).error
    assert "TypeError" in warm.run("features = {'x': object()}", {}).error
    bad = warm.run("features = {", {})
    assert not bad.ok and "SyntaxError" in bad.error


# ----------------------------------------------------------------------------------------------- the pool
def test_the_extractor_is_loaded_once_and_each_request_gets_a_fresh_namespace(warm):
    code = (
        "import os\n"
        "try:\n    seen\nexcept NameError:\n    seen = 0\n"
        "seen += 1\n"
        "features = {'pid': os.getpid(), 'seen': seen}\n"
    )
    res = [warm.run(code, {}) for _ in range(5)]
    assert {r.features["seen"] for r in res} == {1}  # no state leaks between requests
    assert len({r.features["pid"] for r in res}) == 1  # one interpreter served all five
    assert warm.pids() == [res[0].features["pid"]] and warm.stats()["started"] == 1


def test_concurrent_requests_share_the_pool():
    code = "import time\ntime.sleep(0.05)\nfeatures = {'i': trajectory['i']}\n"
    with WarmSandbox(size=3, timeout_s=20) as box:
        out = {}

        def call(i):
            out[i] = box.run(code, {"i": i}).features

        threads = [threading.Thread(target=call, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = box.stats()
    assert out == {i: {"i": i} for i in range(24)}
    assert stats["started"] == 3 and stats["live"] == 3 and stats["requests"] == 24


def test_waiting_for_a_busy_pool_counts_against_the_timeout():
    with WarmSandbox(size=1, timeout_s=20) as box:
        box.run("features = {}", {})
        slow = threading.Thread(target=box.run, args=("import time\ntime.sleep(1.5)\nfeatures = {}", {}))
        slow.start()
        time.sleep(0.2)
        r = box.run("features = {}", {}, timeout_s=0.3)
        slow.join()
    assert not r.ok and "waiting for a free sandbox worker" in r.error


def test_warm_starts_the_pool_and_keeps_it_full(warm):
    code = "import os\nif trajectory.get('die'):\n    os._exit(1)\nfeatures = {'pid': os.getpid()}\n"
    assert not warm.ready
    assert warm.warm(code) == 2 and warm.ready and len(warm.pids()) == 2
    before = set(warm.pids())
    crashed = warm.run(code, {"die": True})
    assert not crashed.ok and crashed.kind == "crash"
    assert warm.ready  # one worker still runs: ready never flaps on a crash
    assert _wait_until(lambda: warm.stats()["live"] == 2)  # replaced in the background, without traffic
    after = set(warm.pids())
    assert len(after) == 2 and after != before and warm.stats()["restarts"] == 1
    assert not warm.stats()["degraded"]


def test_workers_are_recycled_after_max_requests():
    code = "import os\nfeatures = {'pid': os.getpid()}\n"
    with WarmSandbox(size=1, max_requests=2) as box:
        pids = [box.run(code, {}).features["pid"] for _ in range(4)]
        stats = box.stats()
    assert pids[0] == pids[1] != pids[2] == pids[3]
    assert stats["recycled"] == 2 and stats["restarts"] == 0


def test_a_pool_switches_to_new_code(warm):
    assert warm.run("features = {'v': 1}", {}).features == {"v": 1}
    assert warm.run("features = {'v': 2}", {}).features == {"v": 2}
    assert warm.stats()["live"] == 1


def test_close_stops_every_worker_and_later_calls_fail():
    box = WarmSandbox(size=2)
    box.warm("features = {}")
    pids = box.pids()
    box.close()
    assert box.stats()["live"] == 0 and not any(_alive(p) for p in pids)
    later = box.run("features = {}", {})  # fails at once, never raises
    assert not later.ok and later.kind == "unavailable" and "closed" in later.error


def test_workers_stop_when_the_pool_is_garbage_collected():
    box = WarmSandbox(size=1)
    box.run("features = {}", {})
    (pid,) = box.pids()
    del box
    gc.collect()
    assert _wait_until(lambda: not _alive(pid))


@pytest.mark.skipif(not POSIX, reason="checks process liveness with POSIX signals")
def test_workers_exit_when_the_parent_process_dies(tmp_path):
    script = (
        "import os, sys\n"
        "from tabagent.tabschema.warm import WarmSandbox\n"
        "box = WarmSandbox(size=2)\n"
        "box.warm('features = {}')\n"
        "print(*box.pids(), flush=True)\n"
        "os._exit(0)\n"  # no clean-up at all, as on a crash or SIGKILL
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    pids = [int(p) for p in out.stdout.split()]
    assert len(pids) == 2
    assert _wait_until(lambda: not any(_alive(p) for p in pids))


@pytest.mark.skipif(not POSIX, reason="kills a worker with a POSIX signal")
def test_a_worker_killed_while_idle_is_replaced_without_failing_a_request(monkeypatch):
    import signal

    import tabagent.tabschema.warm as warm_module

    code = "import os\nfeatures = {'pid': os.getpid()}\n"

    def kill_the_worker(box):
        (worker,) = box._workers
        os.kill(worker.pid, signal.SIGKILL)
        assert _wait_until(lambda: worker.proc.poll() is not None)

    with WarmSandbox(size=1) as box:
        first = box.run(code, {}).features["pid"]
        kill_the_worker(box)
        second = box.run(code, {})  # found dead when the pool hands it out
        assert second.ok and second.features["pid"] != first and box.stats()["restarts"] == 1
        # a worker that dies right after that check: the request is sent to a fresh worker instead
        monkeypatch.setattr(warm_module._Worker, "alive", lambda self: True)
        kill_the_worker(box)
        third = box.run(code, {})
        assert third.ok and third.features["pid"] not in (first, second.features["pid"])
        assert box.stats()["restarts"] == 2


# ----------------------------------------------------------------------------------------------- review fixes
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="macOS does not enforce RLIMIT_CPU")
def test_code_cannot_lift_its_cpu_limit_past_the_hard_limit():
    lift = (
        "import resource, time\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_CPU)\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))\n"  # as high as it may go
        "t = time.process_time()\n"
        "while time.process_time() - t < 30:\n    pass\n"
        "features = {}\n"
    )
    with WarmSandbox(size=1, timeout_s=60, cpu_s=1) as box:
        t0 = time.monotonic()
        r = box.run(lift, {})
        assert not r.ok and r.kind == "crash" and time.monotonic() - t0 < 15  # killed at the hard limit


def test_a_pool_refills_itself_after_a_failed_start_without_traffic(monkeypatch):
    import tabagent.tabschema.warm as warm_module

    real, calls = warm_module._Worker.__init__, {"n": 0}

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # one transient failure (fork EAGAIN, a slow start under load, ...)
            raise OSError(11, "Resource temporarily unavailable")
        return real(self, *a, **kw)

    monkeypatch.setattr(warm_module._Worker, "__init__", flaky)
    with WarmSandbox(size=3) as box:
        assert box.warm("features = {}") == 2 and box.ready  # ready with min_ready=1 of 3
        stats = box.stats()
        assert stats["degraded"] and stats["start_failures"] == 1
        assert _wait_until(lambda: box.stats()["live"] == 3)  # retried in the background, no request needed
        assert not box.stats()["degraded"]


def test_readiness_does_not_flap_while_requests_crash_their_workers():
    code = "import os\nif trajectory['crash']:\n    os._exit(1)\nfeatures = {'n': 1}\n"
    with WarmSandbox(size=4) as box:
        box.warm(code)
        samples, stop = [], threading.Event()

        def sample():
            while not stop.is_set():
                samples.append(box.ready)
                time.sleep(0.002)

        sampler = threading.Thread(target=sample)
        sampler.start()
        for i in range(60):
            box.run(code, {"crash": i % 10 == 0})  # one decision in ten crashes its worker
            time.sleep(0.005)
        stop.set()
        sampler.join()
    assert samples and all(samples)


def test_an_extractor_that_does_not_compile_fails_loudly_and_fast(caplog):
    from tabagent.tabschema.warm import ExtractorBrokenError

    bad = "features = {'x': trajectory['x']\n"  # SyntaxError
    with WarmSandbox(size=2) as box:
        with pytest.raises(ExtractorBrokenError, match="SyntaxError"):
            box.warm(bad)
        started = box.stats()["started"]
        t0 = time.monotonic()
        results = [box.run(bad, {"x": i}) for i in range(20)]
        assert time.monotonic() - t0 < 0.5  # no interpreter is started per request
        stats = box.stats()
    assert all(r.kind == "broken" and "SyntaxError" in r.error for r in results)
    assert not stats["ready"] and stats["broken"] and stats["started"] == started
    assert stats["start_failures"] >= 1
    assert [r.levelname for r in caplog.records if "cannot run" in r.getMessage()] == ["ERROR"]


def test_the_pool_can_refuse_code_that_breaks_the_extraction_rules():
    with WarmSandbox(size=1, check_code=True) as box:
        r = box.run("import os\nfeatures = {'x': os.getcwd()}\n", {})
        assert not r.ok and r.kind == "broken" and "import: os" in r.error
        assert box.stats()["started"] == 0  # never run
        assert box.run("import json\nfeatures = {'x': json.dumps(1)}\n", {}).ok


def test_a_grandchild_that_keeps_stderr_open_cannot_hang_the_pool():
    escape = (
        "import os, signal, time\n"
        "if os.fork() == 0:\n"  # a grandchild in a new session: outside the worker's process group
        "    os.setsid()\n"
        "    time.sleep(20)\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        "while True:\n    pass\n"
    )
    stay = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"  # a grandchild in the worker's process group
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "print('grandchild', pid, flush=True)\n"
        "os._exit(3)\n"
    )
    with WarmSandbox(size=1, timeout_s=1) as box:
        t0 = time.monotonic()
        r = box.run(escape, {})
        assert not r.ok and r.kind == "timeout" and time.monotonic() - t0 < 6
        r = box.run(stay, {})
        assert not r.ok and r.kind == "crash"
        grandchild = int(r.error.split("grandchild ")[1].split()[0])
        assert _wait_until(lambda: not _alive(grandchild), 5)  # killed with the worker's process group


@pytest.mark.skipif(not POSIX, reason="fork")
@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded:DeprecationWarning")
def test_a_forked_child_starts_its_own_workers():
    code = "import os\nfeatures = {'who': trajectory['who'], 'pid': os.getpid()}\n"
    box = WarmSandbox(size=1)
    box.warm(code)
    parent_pids = set(box.pids())
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child: its inherited worker belongs to the parent
        try:
            results = [box.run(code, {"who": "child"}) for _ in range(20)]
            ok = all(r.ok and r.features["who"] == "child" for r in results)
            fresh = all(r.features["pid"] not in parent_pids for r in results if r.ok)
            os.write(write, b"1" if ok and fresh else b"0")
            box.close()
        finally:
            os._exit(0)
    os.close(write)
    try:
        parent = [box.run(code, {"who": "parent"}) for _ in range(20)]
        assert all(r.ok and r.features["who"] == "parent" for r in parent)
        assert {r.features["pid"] for r in parent} <= parent_pids | set(box.pids())
        os.waitpid(pid, 0)
        assert os.read(read, 1) == b"1"
    finally:
        os.close(read)
        box.close()


@pytest.mark.skipif(not POSIX, reason="checks process liveness with POSIX signals")
def test_workers_exit_when_the_parent_is_killed_in_the_middle_of_a_request():
    script = (
        "import os, signal, threading, time\n"
        "from tabagent.tabschema.warm import WarmSandbox\n"
        "box = WarmSandbox(size=1, timeout_s=60)\n"
        "box.warm('import time\\ntime.sleep(trajectory[\"s\"])\\nfeatures = {}')\n"
        "print(*box.pids(), flush=True)\n"
        "threading.Thread(target=box.run, args=(box._warm_code, {'s': 50})).start()\n"
        "time.sleep(0.5)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"  # no clean-up at all
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    (worker,) = [int(p) for p in out.stdout.split()]
    assert _wait_until(lambda: not _alive(worker), 10)


def test_a_malicious_extractor_cannot_read_the_protocol_state_or_spy_on_later_requests():
    spy = (
        "import json\n"
        "if not hasattr(json, 'seen'):\n"
        "    json.seen = []\n"
        "    real = json.loads\n"
        "    def hook(s, *a, **k):\n"
        "        obj = real(s, *a, **k)\n"
        "        if isinstance(obj, dict) and 'trajectory' in obj:\n"
        "            json.seen.append(obj['trajectory']['user'])\n"
        "        return obj\n"
        "    json.loads = hook\n"
        "features = {'user': trajectory['user'], 'seen': list(json.seen)}\n"
    )
    forge = (
        "import sys\n"
        "m = sys.modules['__main__']\n"
        "found = [k for k in dir(m) if k in ('send', 'token', 'replies')]\n"
        "features = {'n': trajectory['n'], 'found': ','.join(found)}\n"
    )
    builtins_patch = "__builtins__['len'] = lambda x: 42\nfeatures = {'n': len([1])}\n"
    with WarmSandbox(size=1) as box:
        assert [box.run(spy, {"user": u}).features["seen"] for u in ("alice", "bob")] == [[], []]
        assert box.run(forge, {"n": 1}).features == {"n": 1, "found": ""}
        assert box.run(builtins_patch, {}).features == {"n": 42}  # its own copy of the builtins
        assert box.run("features = {'n': len([1, 2])}", {}).features == {"n": 2}


def test_large_replies_and_noisy_stderr_stay_fast_and_tails_are_per_request():
    big = "features = {'x': 'a' * trajectory['n']}\n"
    with WarmSandbox(size=1) as box:
        t0 = time.monotonic()
        r = box.run(big, {"n": 8_000_000})
        assert r.ok and len(r.features["x"]) == 8_000_000 and time.monotonic() - t0 < 5
        noisy = (
            "import os, sys\nsys.stderr.write(trajectory['say'])\nsys.stderr.flush()\n"
            "if trajectory['die']:\n    os._exit(2)\nfeatures = {}\n"
        )
        assert box.run(noisy, {"say": "FIRST REQUEST " * 100, "die": False}).ok
        crash = box.run(noisy, {"say": "SECOND REQUEST", "die": True})
        assert "SECOND REQUEST" in crash.error and "FIRST REQUEST" not in crash.error


def test_closing_a_pool_with_requests_in_flight_fails_them_fast():
    code = "import time\ntime.sleep(trajectory['s'])\nfeatures = {'s': trajectory['s']}\n"
    from concurrent.futures import ThreadPoolExecutor

    for _ in range(5):
        box = WarmSandbox(size=3, timeout_s=10)
        box.warm(code)
        with ThreadPoolExecutor(8) as ex:
            futures = [ex.submit(box.run, code, {"s": 2.0}) for _ in range(6)]
            time.sleep(0.2)
            t0 = time.monotonic()
            box.close(wait_s=0.2)
            results = [f.result(timeout=10) for f in futures]  # nothing raises
            assert time.monotonic() - t0 < 3
        assert all(not r.ok and r.kind == "unavailable" and "closed" in r.error for r in results)
        assert box.stats()["live"] == 0


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_timeouts_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="positive"):
        WarmSandbox(timeout_s=bad)
    with WarmSandbox(size=1) as box, pytest.raises(ValueError, match="positive"):
        box.run("features = {}", {}, timeout_s=bad)


# ----------------------------------------------------------------------------------------------- final review
def test_frame_and_traceback_introspection_is_refused():
    """A generator's frame leads to the worker loop's locals: the checks refuse that route, as attributes
    and as the strings attrgetter would follow."""
    from tabagent.tabschema.checks import check_code
    from tabagent.tabschema.warm import static_refusal

    walk = (
        "def walk():\n    yield gen.gi_frame.f_back\n"
        "gen = walk()\nframe = next(gen)\n"
        "while frame is not None and 'send' not in frame.f_locals:\n    frame = frame.f_back\n"
        "features = {'reached': frame is not None}\n"
    )
    refusal = static_refusal(walk)
    assert refusal and "internals: line 2: .gi_frame" in refusal and ".f_locals" in refusal
    getter = "import operator\nfeatures = {'x': str(operator.attrgetter('gi_frame.f_back')(iter([])))}\n"
    assert "internals" in static_refusal(getter)
    traceback_route = (
        "try:\n    1 / 0\nexcept Exception as e:\n    f = e.__traceback__.tb_frame\nfeatures = {}\n"
    )
    assert {i.kind for i in check_code(traceback_route)} >= {"dunder", "internals"}
    fine = (
        "import json\n"
        "def f(t):\n    return {'n': len(t['steps']), 'code': 'co_' + 'x', 'frame': t['frame']}\n"
        "if __name__ == '__main__':\n    features = f(trajectory)\n"
    )
    assert static_refusal(fine) is None
    with WarmSandbox(size=1, check_code=True) as box:
        r = box.run(walk, {})
        assert r.kind == "broken" and box.stats()["started"] == 0


def test_a_missing_interpreter_is_an_error_result_not_an_exception():
    from tabagent.tabschema.warm import WorkerStartError

    with WarmSandbox(size=2, python="/nonexistent/python3") as box:
        with pytest.raises(WorkerStartError, match="FileNotFoundError"):
            box.warm("features = {}")
        r = box.run("features = {}", {})
        assert not r.ok and r.kind == "unavailable" and "FileNotFoundError" in r.error
        assert box.stats()["start_failures"] >= 3 and not box.ready


@pytest.mark.skipif(not POSIX, reason="a POSIX shell wrapper")
def test_workers_that_keep_dying_while_idle_are_replaced_with_backoff_and_logged(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="tabagent.tabschema.warm")
    wrapper = tmp_path / "dies_soon.sh"  # an environment that kills every worker 0.3 s after it starts
    wrapper.write_text(f'#!/bin/sh\n(sleep 0.3; kill -9 $$) &\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    with WarmSandbox(size=4, python=str(wrapper)) as box:
        assert box.warm("features = {}") == 4
        time.sleep(6)
        stats = box.stats()
    assert 4 <= stats["idle_deaths"] <= 12, stats  # was about 17 in 6 s without backoff
    assert stats["restarts"] >= stats["idle_deaths"]
    died = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "worker_died_idle"]
    assert len(died) == stats["idle_deaths"]
