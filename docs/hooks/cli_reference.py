"""mkdocs hook: the command-line reference, rendered from the argparse parser.

``reference/cli.md`` does not exist in docs/. It is generated at build time from
``tabagent.cli.build_parser``, so the reference lists exactly the commands and flags the installed
version has: the help text, the defaults and the choices all come from the code, and a flag added to the
CLI appears here with no edit."""

from __future__ import annotations

import argparse
import os
import re

from mkdocs.structure.files import File

PAGE = "reference/cli.md"
INTRO = """\
---
title: "Command line"
kind: "Reference"
lead: "Every `tabagent` command and flag, rendered from the argparse parser of the installed version."
---

# Command line

`tabagent` is installed with the package. Each command below shows its usage line, its description and its
options, exactly as `tabagent COMMAND --help` prints them. Add `-v` to any command to log progress and
`-vv` to see tracebacks.

```console
$ tabagent --help
{top}
```

## Exit status

{exit_status}

"""


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _default(action: argparse.Action) -> str:
    if action.default in (None, False, argparse.SUPPRESS) or action.default == []:
        return ""
    return f"`{action.default}`"


def _choices(action: argparse.Action) -> str:
    return ", ".join(f"`{c}`" for c in action.choices) if action.choices else ""


def _cell(text: str | None) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("%(default)s", "")).strip()
    text = text.replace("|", "\\|")
    return re.sub(r"\s*\(default:\s*\)", "", text)


def _options(parser: argparse.ArgumentParser) -> str:
    positional, options = [], []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if action.option_strings:
            flags = ", ".join(f"`{s}`" for s in action.option_strings)
            if action.metavar or (action.nargs != 0 and not isinstance(action, argparse._CountAction)):
                meta = action.metavar or action.dest.upper()
                if action.nargs != 0 and not isinstance(action, argparse._StoreTrueAction):
                    flags += f" `{meta}`" if not action.choices else ""
            options.append((flags, action))
        else:
            positional.append(action)
    out = []
    if positional:
        out += ["| Argument | Description |", "|---|---|"]
        for a in positional:
            out.append(f"| `{a.metavar or a.dest}` | {_cell(a.help)} |")
        out.append("")
    if options:
        out += ["| Option | Description | Choices | Default |", "|---|---|---|---|"]
        for flags, a in options:
            out.append(f"| {flags} | {_cell(a.help)} | {_choices(a)} | {_default(a)} |")
        out.append("")
    return "\n".join(out)


def render() -> str:
    os.environ["COLUMNS"] = "88"
    from tabagent import cli

    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    exit_status = (cli.main.__doc__ or "").split("Under ``-vv``")[0].strip()
    exit_status = re.sub(
        r"\s+", " ", exit_status.replace("Run the CLI; returns the exit status:", "")
    ).strip()
    exit_status = "A command returns " + exit_status.replace("``", "`").rstrip(".") + "."
    text = INTRO.format(top=parser.format_help().strip(), exit_status=exit_status)
    helps = {a.dest: a.help for a in sub._choices_actions}
    for name, p in sub.choices.items():
        text += f"## tabagent {name}\n\n"
        summary = (helps.get(name) or "").strip()
        desc = re.sub(r"[ \t]+\n", "\n", (p.description or "").strip())
        if desc and not desc.lower().startswith(summary.lower()):
            text += (f"{summary[:1].upper()}{summary[1:]}.\n\n" if summary else "") + desc + "\n\n"
        else:
            text += (desc or (f"{summary[:1].upper()}{summary[1:]}." if summary else "")) + "\n\n"
        text += f"```console\n{p.format_usage().strip()}\n```\n\n"
        text += _options(p) + "\n"
    return text


def on_files(files, config):
    files.append(File.generated(config, PAGE, content=render()))
    return files
