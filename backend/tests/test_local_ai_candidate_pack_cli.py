"""The benchmark-only candidate CLI cannot become a public release path."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_candidate_install_explicitly_skips_release_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    captured: dict[str, object] = {}

    async def run(action: str, **kwargs: object) -> int:
        captured["action"] = action
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(candidate_cli, "_run_lifecycle", run)
    assert await candidate_cli.execute("install") == 0
    assert captured["action"] == "install"
    assert captured["require_release"] is False


def test_candidate_cli_normalizes_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.local_ai_candidate_pack as candidate_cli

    async def interrupt(_action: str) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(candidate_cli, "execute", interrupt)

    assert candidate_cli.main(["install"]) == 130
    assert (
        capsys.readouterr().err == "ERROR: candidate model pack command interrupted.\n"
    )
