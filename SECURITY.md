# Security policy

TabSchema executes code that an LLM writes from your agent's logs. By default it runs that code in separate,
resource-limited Python processes with an empty environment, which keeps API keys out of reach but is **not a
security boundary**. Review the generated extractor (`extract.py`) before you deploy it, and use
`DockerSandbox` (no network, read-only filesystem; requires Docker) when the logs or the code are untrusted:

- serving a head: `tabagent serve my_head --sandbox docker`, or `sandbox="docker"` in `tabagent.load`,
  `TabAgentHead.load` and `ToolFilter` (`"docker:<image>"` picks the image);
- onboarding: `tabagent.onboard(..., sandbox="docker")` runs discovery, the decision table and the returned head
  in Docker;
- discovery: `TabSchema(llm, sandbox=DockerSandbox())` or `tabagent.cross_validate_discovery(...,
  sandbox=DockerSandbox())`, with `DockerSandbox` from `tabagent.tabschema.sandbox`.

`--trust-extractor` (`trust_extractor=True`) runs a reviewed extractor in-process, outside any sandbox.

## What the default sandboxes do

Onboarding runs the code in `SubprocessSandbox`: a fresh `python -I -S` interpreter per batch of
trajectories. A served head runs it in `WarmSandbox`: a pool of the same interpreters that start once and
then each run the extractor on one prefix per request. Both:

- start the interpreter in isolated mode without site packages (`-I -S`), in a temporary directory, with an
  empty environment, so the code cannot read API keys or credentials from environment variables;
- limit the file size the code can write and disable core dumps, and on Linux also limit its address space
  (`mem_mb`, 2 GB by default) and its CPU time (`cpu_s` per batch, or per request in a warm worker, under a
  hard limit the code cannot raise). macOS accepts these two limits but does not enforce them (checked on
  macOS 26: code ran three times past a `cpu_s` that `SubprocessSandbox` set as both soft and hard limit);
- stop code that runs past its timeout with an alarm, and kill an interpreter that ignores the alarm (a warm
  worker with its whole process group);
- keep the code's output (`print`, writes to stdout) off the result channel, and give it an empty stdin.

The deterministic checks of the compile step refuse code that imports modules outside a small standard-library
list, calls `open`, `eval`, `exec` and similar builtins, or reaches for interpreter internals. A head served
from a sandbox name (`warm`, `subprocess`, `docker`) re-applies these checks to `extract.py` before running it,
and refuses code that fails them (every decision is then deferred). A sandbox instance you pass runs what it
is given. `extract.py` must match the SHA-256 that the card records, and the card is covered by the artifact's
manifest.

**What a warm worker adds.** A worker serves many requests, one after another, in one interpreter. A buggy
extractor is contained: each request runs in a fresh namespace with its own copy of the builtins, and the
protocol keeps its state out of the code's reach (no token or reply function in any module's globals, private
references to the JSON functions it uses). The static checks also refuse the known routes to other code's
frames (dunder attributes, and `gi_frame`, `f_back`, `f_locals`, `tb_frame` and the like, also as strings).
A malicious extractor is still not contained. With only the allowed imports it can leave code behind in a
shared standard-library module (`json`, `re`, ...) that its own later runs call. It can then **observe the
prefixes of later requests and change the answers to them**. Static checks over a dynamic language cannot
rule out every route, and a worker is only replaced after a timeout, a crash, 5000 requests or its CPU budget.
Serve a head whose extractor you have not reviewed with `--sandbox subprocess` (a fresh interpreter per
decision: no state survives a decision) or `--sandbox docker`, and review `extract.py` before you serve it
with the warm default. `--allow-unchecked-extractor` runs an extractor that fails the checks (a head saved by
an earlier version); use it only after reviewing the code.

None of this stops a determined attacker: the code still runs as your user, on your machine, with network
access (the checks forbid network modules, but they are static checks). Code or logs you do not trust belong in
`DockerSandbox`.

## Serving

- `tabagent serve` has no authentication and no TLS. It binds to `127.0.0.1` by default; bind it elsewhere
  only on a private network or behind an authenticated proxy, never on the public internet.
- Logs carry no request content by default: no prefix, run, candidates, answer or full error message (an
  extractor's error can quote the prefix). The extractor failure class and the exception name are logged
  instead, and stack frames of internal errors without their messages. `--log-content` adds the content, so
  use it only where the logs may hold your agent's data.
- `/metrics` and `/drift` expose feature names and statistics, `/card` the feature card, and `/version` the
  hashes of the served artifact; none of them exposes request content. `model/feature_stats.json`, like
  `featurizer.json`, holds values seen in training: categories (candidate names, for example), numeric
  quantiles and text lengths, never whole texts.
- `--redact module:function` applies the head's redaction function to every `run` before its prefix is built.

## Reporting a vulnerability

Please report a vulnerability privately through the repository's **Security → Report a vulnerability** form
rather than in a public issue. We will acknowledge the report and agree on a disclosure timeline with you.
