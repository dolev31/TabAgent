"""Fabricated mini-logs in each supported format (no real agent data)."""

import json

TOOLS = {
    "search_flights": ("search flights between two airports", ["origin", "destination", "date"]),
    "book_flight": ("book a seat on a flight", ["flight_id", "passenger"]),
    "get_weather": ("weather forecast for a city", ["city"]),
    "send_email": ("send an email message", ["to", "body"]),
    "read_inbox": ("read the latest emails", ["limit"]),
    "play_song": ("play music", ["query"]),
}
TASKS = [
    ("find a flight to {c} and book it", ["search_flights", "book_flight"]),
    ("what's the weather in {c}", ["get_weather"]),
    ("reply to the email about {c}", ["read_inbox", "send_email"]),
    ("play a song from {c}", ["play_song"]),
]
CITIES = ["rome", "oslo", "lima"]
SYSTEM = "You are a helpful travel and email assistant."


def openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": t,
                "description": d,
                "parameters": {"type": "object", "properties": {a: {"type": "string"} for a in args}},
            },
        }
        for t, (d, args) in TOOLS.items()
    ]


def messages_api_tools():
    return [
        {"name": t, "description": d, "input_schema": {"type": "object", "properties": {a: {} for a in args}}}
        for t, (d, args) in TOOLS.items()
    ]


def task(i):
    intent, used = TASKS[i % len(TASKS)]
    return intent.format(c=CITIES[i % len(CITIES)]), used


def _args(tool):
    return {a: f"{a}-value" for a in TOOLS[tool][1]}


def openai_messages(i):
    intent, used = task(i)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": intent},
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "yes please"},
    ]
    for k, t in enumerate(used):
        call = {"id": f"c{k}", "type": "function", "function": {"name": t, "arguments": json.dumps(_args(t))}}
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": f"result of {t}: ok"},
        ]
    return msgs + [{"role": "assistant", "content": f"FINAL ANSWER: done ({intent})"}]


def messages_api_run(i):
    intent, used = task(i)
    msgs = [
        {"role": "user", "content": intent},
        {"role": "assistant", "content": [{"type": "text", "text": "Let me check."}]},
        {"role": "user", "content": [{"type": "text", "text": "yes please"}]},
    ]
    for k, t in enumerate(used):
        msgs += [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"tu{k}", "name": t, "input": _args(t)}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"tu{k}", "content": f"result of {t}: ok"}
                ],
            },
        ]
    msgs.append(
        {"role": "assistant", "content": [{"type": "text", "text": f"FINAL ANSWER: done ({intent})"}]}
    )
    return {"system": SYSTEM, "messages": msgs, "tools": messages_api_tools()}


def langchain_run(i):
    intent, used = task(i)
    msgs = [
        {"type": "system", "content": SYSTEM},
        {"type": "human", "content": intent},
        {"type": "ai", "content": "Let me check.", "tool_calls": []},
        {"type": "human", "content": "yes please"},
    ]
    for k, t in enumerate(used):
        msgs += [
            {"type": "ai", "content": "", "tool_calls": [{"name": t, "args": _args(t), "id": f"lc{k}"}]},
            {"type": "tool", "content": f"result of {t}: ok", "tool_call_id": f"lc{k}", "name": t},
        ]
    msgs.append({"type": "ai", "content": f"FINAL ANSWER: done ({intent})"})
    return {"messages": msgs}


def openai_run(i, outcome=None, tools=True, **extra):
    run = {"task_id": f"task{i % len(TASKS)}", "messages": openai_messages(i), **extra}
    if tools:
        run["tools"] = openai_tools()
    run.update(outcome if outcome is not None else {"reward": 1.0})
    return run


def write_openai_logs(d, n=24, fail_every=8):
    """n runs, one per file; every ``fail_every``-th run failed (reward 0)."""
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        reward = 0.0 if fail_every and i % fail_every == fail_every - 1 else 1.0
        (d / f"run{i:02d}.json").write_text(json.dumps(openai_run(i, {"reward": reward})))
    return d


# --------------------------------------------------------------------------- tau2-bench
def tau2_messages(tools, user_tool=None):
    msgs = [
        {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None, "turn_idx": 0},
        {"role": "user", "content": "I need help with my booking.", "tool_calls": None, "turn_idx": 1},
    ]
    for k, t in enumerate(tools):
        cid = f"call_{k}"
        msgs += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": cid, "name": t, "arguments": {"user_id": "u1"}, "requestor": "assistant"}
                ],
                "raw_data": {"message": {"content": "RAW PAYLOAD"}},
            },
            {"id": cid, "role": "tool", "content": f"{t} output", "requestor": "assistant", "error": False},
        ]
        if user_tool and k == 0:
            msgs += [
                {
                    "role": "user",
                    "content": None,
                    "tool_calls": [{"id": "u0", "name": user_tool, "arguments": {}, "requestor": "user"}],
                },
                {"id": "u0", "role": "tool", "content": "done", "requestor": "user", "error": False},
            ]
    return msgs + [{"role": "assistant", "content": "Your booking is updated.", "tool_calls": None}]


def tau2_results(domain, sims, implementation="llm_agent", tool_defs=None):
    """sims: list of (task_id, tools called, reward or None)."""
    tasks = sorted({s[0] for s in sims})
    return {
        "timestamp": "2025-01-01T00:00:00",
        "info": {
            "num_trials": 1,
            "agent_info": {"implementation": implementation, "llm": "agent-model"},
            "user_info": {"implementation": "user_simulator", "llm": "user-model"},
            "environment_info": {
                "domain_name": domain,
                "policy": f"# {domain} policy",
                "tool_defs": tool_defs,
            },
        },
        "tasks": [
            {
                "id": t,
                "ticket": f"TICKET for task {t}",
                "user_scenario": {"instructions": {"reason_for_call": "SECRET SCENARIO"}},
                "evaluation_criteria": {"actions": [{"name": "gold_only_tool", "arguments": {}}]},
            }
            for t in tasks
        ],
        "simulations": [
            {
                "id": f"sim-{k}",
                "task_id": t,
                "trial": 0,
                "messages": tau2_messages(called, user_tool="toggle_data" if domain == "telecom" else None),
                "reward_info": None if r is None else {"reward": r, "db_check": {"db_match": r == 1.0}},
                "termination_reason": "user_stop",
            }
            for k, (t, called, r) in enumerate(sims)
        ],
    }


# --------------------------------------------------------------------------- CUGA
API_DOCS = {
    "phone": {
        "phone_search_contacts": ("Search your contacts.", ["query", "relationship"]),
        "phone_show_text_messages": ("Show text messages with a contact.", ["phone_number"]),
        "phone_delete_alarm": ("Delete an alarm.", ["alarm_id"]),
    },
    "venmo": {
        "venmo_send_money": ("Send money to a user.", ["receiver_email", "amount"]),
        "venmo_show_balance": ("Show your balance.", []),
    },
}


def cuga_api_docs(app):
    return {
        api: {
            "app_name": app,
            "secure": True,
            "api_name": api,
            "path": f"/{api}",
            "method": "GET",
            "description": f'description: {d}\nmodel: Model\n---\nsuccess:\n{{"message": "string"}}',
            "parameters": [{"name": a, "type": "string", "required": True} for a in args],
            "response_schemas": {"success": {"message": "string"}},
        }
        for api, (d, args) in API_DOCS[app].items()
    }


def _prompt_block(app):
    return f'`Available APIs`:\n"""\n{json.dumps({app: cuga_api_docs(app)}, indent=2)}\n"""'


def _round(app, intent, offered, calls, old_format, with_block):
    planner = {
        "thoughts": [f"we need {app}"],
        "action": "ApiShortlistingAgent",
        "action_input_shortlisting_agent": {"app_name": app, "task_description": f"find APIs to {intent}"},
    }
    result = {"thoughts": ["ranked"], "result": [{"name": a, "relevance_score": 0.9} for a in offered]}
    human = f"`User Intent`: **Input task**: find APIs\n\n`Current Application`: {app}\n\n"
    human += _prompt_block(app) if with_block else "`Available APIs`: (omitted)"
    if old_format:  # the original convert2tabular layout: [system+human, generation]
        planner_prompts = [
            {
                "role": "system",
                "value": f'System: plan.\nHuman: Summary:\n"""\nnone yet\n"""\n`USER_GOAL`: `{intent}`.',
            },
            {"role": "generation", "value": json.dumps(planner)},
        ]
        short_prompts = [
            {"role": "system", "value": f"System: You select APIs.\nHuman: {human}"},
            {"role": "generation", "value": json.dumps(result)},
        ]
    else:
        planner_prompts = [
            {"role": "system", "value": "You plan API calls."},
            {"role": "human", "value": f"`USER_GOAL`: `{intent}`."},
            {"role": "assistant", "value": "```json\n" + json.dumps(planner) + "\n```"},
        ]
        short_prompts = [
            {"role": "system", "value": "You select APIs."},
            {"role": "human", "value": human},
            {"role": "assistant", "value": json.dumps(result)},
        ]
    code = "\n".join(
        f'r{k} = await call_api(\n    "{app}",\n    "{a}",\n    {{}})' for k, a in enumerate(calls)
    )
    return [
        {
            "name": "APIPlannerAgent",
            "prompts": planner_prompts,
            "data": "" if old_format else json.dumps(planner),
        },
        {
            "name": "ShortlisterAgent",
            "prompts": short_prompts,
            "data": "" if old_format else json.dumps(result),
        },
        {"name": "APIPlannerAgent", "prompts": [], "data": json.dumps({"action": "CoderAgent"})},
        {"name": "CodeAgent", "prompts": [], "data": json.dumps({"code": code, "execution_output": "ok"})},
        {"name": "api_call", "data": json.dumps({"app_name": app, "function_name": calls[0], "args": {}})},
        {"name": "api_response", "data": json.dumps({"message": "ok"})},
    ]


def cuga_trajectory(
    task_id, rounds, score=1.0, old_format=False, with_block=True, intent="pay my roommate back"
):
    """rounds: list of (app, offered APIs, APIs called as written in call_api)."""
    steps = [
        {"name": "TaskAnalyzerAgent", "prompts": [], "data": json.dumps({"apps": [r[0] for r in rounds]})},
        {"name": "PlanControllerAgent", "prompts": [], "data": json.dumps({"next_subtask": "use the apps"})},
    ]
    for app, offered, calls in rounds:
        steps += _round(app, intent, offered, calls, old_format, with_block)
    steps += [
        {"name": "FinalAnswerAgent", "prompts": [], "data": json.dumps({"final_answer": "FINAL: all done"})},
        {"name": "EvaluationResult", "prompts": [], "data": json.dumps({"score": score})},
    ]
    return {
        "intent": intent,
        "dataset_name": "fabricated",
        "actions_count": len(steps),
        "task_id": task_id,
        "eval": {"passed": score > 0},
        "steps": steps,
        "score": score,
    }
