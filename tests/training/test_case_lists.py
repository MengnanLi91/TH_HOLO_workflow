from __future__ import annotations

import pytest

from training.case_lists import load_case_selection


def test_case_selection_reads_newline_file(tmp_path):
    path = tmp_path / "heldout.txt"
    path.write_text("case_a\n\ncase_b\n", encoding="utf-8")

    assert load_case_selection([], path, label="exclude_cases") == ["case_a", "case_b"]


def test_case_selection_rejects_inline_and_file(tmp_path):
    path = tmp_path / "heldout.txt"
    path.write_text("case_b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only one"):
        load_case_selection(["case_a"], path, label="force_test")


@pytest.mark.parametrize("contents", ["", "case_a\ncase_a\n"])
def test_case_selection_rejects_invalid_file(tmp_path, contents):
    path = tmp_path / "heldout.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_case_selection([], path, label="exclude_cases")
