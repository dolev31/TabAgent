"""Trajectory adapters: read an agent's native logs into TabSchema trajectories and decision points.

An adapter implements ``load(path) -> list[Trajectory]``, ``decision_points(traj, target)``,
``candidates(dp)``, ``labels(dp)`` and ``components(trajs)``, and exposes ``outcome_keys``. A decision
point's ``prefix`` holds only what was known before the decision (``{<task fields>, "steps": [{"name": ...,
...}]}``); ``labels`` may read the full run (for tool shortlisting, the tools the successful run used).

* ``OpenAIMessagesAdapter``: chat-message runs (OpenAI, Anthropic Messages API format, LangChain dumps);
* ``Tau2Adapter``: τ²-bench results files;
* ``CugaAdapter``: CUGA agent trajectories (the API shortlister built in);
* ``GenericJsonAdapter``: any log with named steps, with your own candidates and labels;
* ``detect_adapter(path)``: picks one of the above from the first files.
"""

from tabagent.adapters.common import parse_catalog
from tabagent.adapters.cuga import SHORTLISTER, CugaAdapter
from tabagent.adapters.detect import detect_adapter, detect_format
from tabagent.adapters.openai_messages import (
    NEXT_TOOL_CALL,
    TOOL_SHORTLIST,
    OpenAIMessagesAdapter,
    normalize_messages,
)
from tabagent.adapters.tau2 import Tau2Adapter
from tabagent.tabschema.trajectory import GenericJsonAdapter

__all__ = [
    "NEXT_TOOL_CALL",
    "SHORTLISTER",
    "TOOL_SHORTLIST",
    "CugaAdapter",
    "GenericJsonAdapter",
    "OpenAIMessagesAdapter",
    "Tau2Adapter",
    "detect_adapter",
    "detect_format",
    "normalize_messages",
    "parse_catalog",
]
