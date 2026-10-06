"""Trajectory adapters: read an agent's native logs into TabSchema trajectories and decision points.

An adapter implements ``load(path) -> list[Trajectory]``, ``decision_points(traj, target)``,
``candidates(dp)``, ``labels(dp)`` and ``components(trajs)``, and exposes ``outcome_keys``. A decision
point's ``prefix`` holds only what was known before the decision (``{<task fields>, "steps": [{"name": ...,
...}]}``); ``labels`` may read the full run (for tool shortlisting, the tools the successful run used).

For ``tabagent.scan`` an adapter may also implement ``llm_calls(traj) -> list[LLMCall]``: every LLM call of a
run, with what it chose, what it was offered, the free text it wrote and the tokens, latency and cost the log
records (``read_usage`` reads the common shapes), plus ``scan_notes`` on how its logs record them. All the
adapters below implement it.

* ``OpenAIMessagesAdapter``: chat-message runs (OpenAI, Anthropic Messages API format, LangChain dumps);
* ``Tau2Adapter``: τ²-bench results files;
* ``CugaAdapter``: CUGA agent trajectories (the API shortlister built in);
* ``GenericJsonAdapter``: any log with named steps, with your own candidates and labels;
* ``detect_adapter(path)``: picks one of the above from the first files.

Every adapter reads files through ``tabagent.adapters.logfiles``: encodings, damaged JSON and JSON Lines,
linked folders, values nested too deeply, and large inputs read again from their files instead of held in
memory. Whatever is skipped or repaired is reported in the adapter's ``warnings``, runs of another log format
included (``detect_adapter(..., on_other_formats="error")`` refuses them with ``MixedFormatsError``).
``tabagent.check_logs`` (CLI ``tabagent check-logs``) diagnoses a log folder before an onboarding.

At decision time, ``live_prefix(adapter, run, target)`` builds a live run's prefix exactly as in training, and
``describe_adapter`` / ``adapter_from_description`` record and rebuild the adapter a head was trained with.
"""

from tabagent.adapters.common import parse_catalog
from tabagent.adapters.cuga import SHORTLISTER, CugaAdapter
from tabagent.adapters.detect import detect_adapter, detect_format
from tabagent.adapters.live import describe as describe_adapter
from tabagent.adapters.live import from_description as adapter_from_description
from tabagent.adapters.live import live_prefix, run_before
from tabagent.adapters.logfiles import LogChangedError, MixedFormatsError
from tabagent.adapters.openai_messages import (
    NEXT_TOOL_CALL,
    TEXT_REPLY,
    TOOL_SHORTLIST,
    OpenAIMessagesAdapter,
    normalize_messages,
)
from tabagent.adapters.tau2 import Tau2Adapter
from tabagent.tabschema.trajectory import GenericJsonAdapter, LLMCall, output_fields, read_usage

__all__ = [
    "NEXT_TOOL_CALL",
    "SHORTLISTER",
    "TEXT_REPLY",
    "TOOL_SHORTLIST",
    "CugaAdapter",
    "GenericJsonAdapter",
    "LLMCall",
    "LogChangedError",
    "MixedFormatsError",
    "OpenAIMessagesAdapter",
    "Tau2Adapter",
    "adapter_from_description",
    "describe_adapter",
    "detect_adapter",
    "detect_format",
    "live_prefix",
    "normalize_messages",
    "output_fields",
    "parse_catalog",
    "read_usage",
    "run_before",
]
