---
title: "Installation"
description: "Install TabAgent with pip, choose the extras you need, set the LLM key TabSchema calls, and check the install."
kind: "Get started"
lead: "One package, Python 3.10 to 3.13, and an extra for each thing you add."
---

# Installation

```bash
pip install tabagent
```

The core package holds the decision table, TabSynth, TabHead (CatBoost), the evaluation protocol, the
scikit-learn estimators and the command line. It depends on numpy, pandas, scikit-learn, catboost and pyyaml.
TabAgent is Apache-2.0 licensed and needs Python 3.10 to 3.13.

## Extras

Add an extra for what you use:

| Extra | Install | Adds |
|---|---|---|
| `embed` | `pip install "tabagent[embed]"` | the e5 context-candidate similarity features (torch and transformers). The encoder, `intfloat/e5-small-v2`, is downloaded from Hugging Face once, about 128 MB. |
| `tabschema` | `pip install "tabagent[tabschema]"` | onboarding from agent logs with OpenAI, Azure OpenAI and any OpenAI-compatible server. |
| `llm` | `pip install "tabagent[llm]"` | Anthropic, Gemini and other providers for TabSchema, through LiteLLM. |
| `serve` | `pip install "tabagent[serve]"` | the HTTP decision service, `tabagent serve` (FastAPI and uvicorn). |
| `hub` | `pip install "tabagent[hub]"` | `tabagent.load("hf://OWNER/REPO")` for heads on the Hugging Face Hub. |
| `all` | `pip install "tabagent[all]"` | every extra above. |

!!! note "What needs which extra"

    The [quickstart](quickstart.md) needs no extra. Onboarding an agent from its logs needs `tabschema` (or `llm`).
    Serving needs `serve`. Reproducing the paper's numbers needs `embed`.

## The LLM TabSchema calls

Only onboarding calls an LLM. `tabagent.onboard` builds it from the model name and the environment:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=...
    ```

=== "OpenAI-compatible server"

    ```bash
    export OPENAI_BASE_URL=http://localhost:8000/v1   # a server that needs no key (vLLM, Ollama) needs no key here
    ```

=== "Azure OpenAI"

    ```bash
    export AZURE_OPENAI_ENDPOINT=...
    export AZURE_OPENAI_API_KEY=...
    ```

=== "Anthropic, Gemini and others"

    ```bash
    pip install "tabagent[llm]"
    export ANTHROPIC_API_KEY=...        # or GEMINI_API_KEY, MISTRAL_API_KEY
    # then: tabagent onboard ... --model anthropic/<model>
    ```

The order in which the environment is read, per-stage models, and the privacy controls are in
[Onboard from logs](../guides/onboard-from-logs.md#llm-providers).

## From source

To get the exact dependency versions the paper's numbers were checked against (pinned in `uv.lock`):

```bash
git clone https://github.com/dolev31/TabAgent && cd TabAgent
uv sync --extra embed
```

Or with pip: `pip install -e ".[embed]"`. pip resolves the newest dependency releases rather than the locked ones.

## Check the install

```console
$ tabagent --version
$ python -c "import tabagent; print(tabagent.__version__)"
```

Then run the [quickstart](quickstart.md).
