# Security policy

TabSchema executes code that an LLM writes from your agent's logs. By default it runs that code in a separate,
resource-limited Python process with an empty environment (`SubprocessSandbox`), which keeps API keys out of
reach but is **not a security boundary**. Review the generated extractor (`extract.py`) before you deploy it,
and use `DockerSandbox` (no network, read-only filesystem; requires Docker) when the logs or the code are
untrusted:

- serving a head: `tabagent serve my_head --sandbox docker`, or `sandbox="docker"` in `tabagent.load`,
  `TabAgentHead.load` and `ToolFilter` (`"docker:<image>"` picks the image);
- discovery: `TabSchema(llm, sandbox=DockerSandbox())` or `tabagent.cross_validate_discovery(...,
  sandbox=DockerSandbox())`, with `DockerSandbox` from `tabagent.tabschema.sandbox`.

`--trust-extractor` (`trust_extractor=True`) runs a reviewed extractor in-process, outside any sandbox.

Please report a vulnerability privately through the repository's **Security → Report a vulnerability** form
rather than in a public issue. We will acknowledge the report and agree on a disclosure timeline with you.
