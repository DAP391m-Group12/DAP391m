from __future__ import annotations

import re
from pathlib import Path

from src.dataset_b.cli import build_parser
from src.utils.path import PROJECT_ROOT


REPRODUCE = PROJECT_ROOT / "REPRODUCE.md"
ACCEPTANCE = PROJECT_ROOT / "docs" / "handbook" / "REPRODUCIBILITY_ACCEPTANCE.md"
HANDBOOK = PROJECT_ROOT / "docs" / "handbook" / "README.md"

ALLOWED_STATUSES = {
    "PASS",
    "PASS_WITH_LIMITATION",
    "DOCUMENTED_NOT_RERUN",
    "NOT_VERIFIED",
    "NOT_APPLICABLE",
}

PUBLIC_COMMANDS = {
    "status",
    "validate-config",
    "validate-artifacts",
    "validate-notebooks",
    "report-index",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reproduce_has_exact_rubric_headings_in_order() -> None:
    text = _read(REPRODUCE)
    headings = re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)
    expected_titles = [
        "General information",
        "Quick Start",
        "Repository structure",
        "Hardware and software",
        "Environment installation",
        "Data artifacts and checkpoints",
        "Configuration and hyperparameters",
        "Training",
        "Evaluation and inference",
        "Seeds and determinism",
        "Result to report mapping",
        "Limitations and troubleshooting",
        "License citation and AI declaration",
    ]

    assert [int(number) for number, _ in headings] == list(range(13))
    assert [title for _, title in headings] == expected_titles
    assert len(headings) == 13


def test_submission_and_reviewed_commit_identities_are_exact() -> None:
    combined = _read(REPRODUCE) + _read(ACCEPTANCE)

    assert "369fe494506fce888d6c917f504d086a1dfcb097" in combined
    assert "ed1afbf783db502ade5dfb959738df51cdb6a5e7" in combined


def test_five_public_cli_commands_are_documented_once_as_the_complete_set() -> None:
    text = _read(REPRODUCE)
    documented = set(re.findall(r"src\.dataset_b\.cli ([a-z-]+)", text))

    assert documented == PUBLIC_COMMANDS


def test_internal_markdown_file_links_exist() -> None:
    for source in (REPRODUCE, ACCEPTANCE, HANDBOOK, PROJECT_ROOT / "README.md"):
        text = _read(source)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            file_target = target.split("#", 1)[0]
            if not file_target:
                continue
            resolved = (source.parent / file_target).resolve()
            assert resolved.exists(), f"Broken link in {source}: {target}"


def test_no_unresolved_placeholder_tokens_remain() -> None:
    combined = _read(REPRODUCE) + _read(ACCEPTANCE)
    forbidden = ("TODO", "TBD", "FIXME", "CHANGEME", "<run_id>", "<stage>")

    assert not any(token in combined for token in forbidden)


def test_acceptance_matrix_uses_only_allowed_status_vocabulary() -> None:
    text = _read(ACCEPTANCE)
    matrix = text.split("## Rubric 0 through 12 matrix", 1)[1].split(
        "## Verification gates", 1
    )[0]
    statuses = re.findall(r"\| `([A-Z_]+)` \|", matrix)

    assert len(statuses) == 13
    assert set(statuses) <= ALLOWED_STATUSES


def test_documentation_does_not_overclaim_prohibited_runtime_results() -> None:
    combined = _read(REPRODUCE) + _read(ACCEPTANCE)
    normalized = re.sub(r"\s+", " ", combined)

    assert "FULL_REPRODUCIBILITY_PASS" not in combined
    assert (
        "No clean installation, model execution, training, or GPU walkthrough "
        "was performed in this documentation task."
    ) in normalized
    assert "clean-host installation was attempted or accepted" in normalized


def test_handbook_routes_to_acceptance_without_replacing_status() -> None:
    text = _read(HANDBOOK)

    assert "REPRODUCIBILITY_ACCEPTANCE.md" in text
    assert "STATUS](STATUS.md) is the sole source for current execution status" in text


def test_navigation_cli_has_no_model_facing_commands() -> None:
    parser = build_parser()
    actions = [
        action
        for action in parser._actions  # noqa: SLF001 - static parser contract test
        if getattr(action, "choices", None)
    ]
    choices = set(actions[0].choices)

    assert choices == PUBLIC_COMMANDS
    assert choices.isdisjoint({"train", "evaluate", "predict", "inference", "serve"})
