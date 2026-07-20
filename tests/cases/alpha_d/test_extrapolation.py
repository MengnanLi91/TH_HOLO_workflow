import pytest

from cases.alpha_d.extrapolation import (
    AXES,
    build_indist_panel,
    build_split,
    parse_case_params,
)


def test_parse_case_params():
    c = parse_case_params("Re_43938__Dr_0p522__Lr_0p073")
    assert round(c.Re) == 43938
    assert c.Dr == pytest.approx(0.522)
    assert c.Lr == pytest.approx(0.073)
    assert c.name == "Re_43938__Dr_0p522__Lr_0p073"


def _mini_grid():
    names = []
    for re in (5000, 43938, 250000):
        for dr in ("0p05", "0p144", "0p239", "0p9"):
            names.append(f"Re_{re}__Dr_{dr}__Lr_0p073")
    return [parse_case_params(n) for n in names]


def _rich_grid():
    names = []
    for re in (5000, 7722, 11927, 18420, 28449, 43938):
        for dr in ("0p144", "0p239", "0p333", "0p428", "0p522", "0p617", "0p711"):
            for lr in ("0p01", "0p031", "0p052", "0p073", "0p094", "0p116"):
                names.append(f"Re_{re}__Dr_{dr}__Lr_{lr}")
    return [parse_case_params(n) for n in names]


def test_build_split_re_high_respects_floor():
    cases = _mini_grid()
    res = build_split(
        cases, axis="Re", side="high", k=1, dr_floor=0.333, exclude_below_dr=0.1
    )
    assert res.shell_axis_values == [250000.0]
    assert all(c.Re == 250000 and c.Dr >= 0.333 for c in res.shell)
    assert all(c.Dr >= 0.333 and c.Re != 250000 for c in res.interior)
    shell_names = {c.name for c in res.shell}
    inter_names = {c.name for c in res.interior}
    assert shell_names.isdisjoint(inter_names)


def test_build_split_dr_low_reaches_below_floor():
    cases = _mini_grid()
    res = build_split(
        cases, axis="Dr", side="low", k=2, dr_floor=0.333, exclude_below_dr=0.1
    )
    assert res.shell_axis_values == pytest.approx([0.144, 0.239])
    assert all(c.Dr in (0.144, 0.239) for c in res.shell)
    assert all(c.Dr >= 0.333 for c in res.interior)
    assert all(c.Dr != 0.05 for c in res.shell + res.interior)


def test_build_split_is_single_source_for_symmetry():
    cases = _mini_grid()
    a = build_split(cases, axis="Re", side="high", k=1)
    b = build_split(cases, axis="Re", side="high", k=1)
    assert [c.name for c in a.shell] == [c.name for c in b.shell]


def test_axes_constant():
    assert AXES == ("Re", "Dr", "Lr")


def test_guarded_dr_split_holds_out_full_shell_but_reports_inner_re_lr():
    cases = _rich_grid()
    res = build_split(
        cases,
        axis="Dr",
        side="low",
        k=2,
        report_guard_axes=("Re", "Lr"),
        report_guard_k=1,
    )

    assert {c.Dr for c in res.shell} == {0.144, 0.239}
    assert {c.Dr for c in res.report} == {0.144, 0.239}
    assert {c.name for c in res.report}.issubset({c.name for c in res.shell})
    assert any(c.Re == 5000 for c in res.shell)
    assert all(c.Re not in {5000, 43938} for c in res.report)
    assert all(c.Lr not in {0.01, 0.116} for c in res.report)
    assert all(c.Dr < 0.333 for c in res.report)


def test_build_indist_panel_is_deterministic_and_inside_common_box():
    cases = _rich_grid()
    first = build_indist_panel(cases, count=12, guard_k=1)
    second = build_indist_panel(cases, count=12, guard_k=1)

    assert [c.name for c in first] == [c.name for c in second]
    assert len(first) == 12
    assert all(c.Dr >= 0.333 for c in first)
    assert all(c.Re not in {5000, 43938} for c in first)
    assert all(c.Dr not in {0.333, 0.711} for c in first)
    assert all(c.Lr not in {0.01, 0.116} for c in first)
