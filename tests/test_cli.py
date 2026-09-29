import pytest

from tabagent.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("components", "onboard", "serve", "datasets"):
        assert cmd in out


def test_onboard_requires_logs():
    with pytest.raises(SystemExit):
        main(["onboard"])
