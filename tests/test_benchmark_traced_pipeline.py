"""The pipeline comparison, checked against known observation sets.

`improvement_factor` is the headline this file has produced twice, and the
number the gate reads. It is one division over two wall times — and until
now it could only run on a machine with an MQTT broker and Redis, so no
test could ask it what it does with a comparison that is not one.

The first version of this benchmark appended a constant for run 2's lookup
stage, which is how "20.5 ms -> 0.05 ms" came to be reported as measured.
The arithmetic was never the problem; the absence of anywhere to check it
was.
"""
import pytest

from research.benchmark_traced_pipeline import percentile, summarise


def _run(name, wall_s, *, cases=132, frames=None):
    return {"run": name, "wall_s": wall_s, "cache_hits": 0,
            "cases": cases,
            "serialised_frames_valid": cases if frames is None else frames,
            "stage_p50_ms": {"cache": 0.1, "lookup": 20.5},
            "stage_p95_ms": {"cache": 0.2, "lookup": 21.0}}


def test_the_improvement_factor_is_the_ratio_of_the_two_wall_times():
    report = summarise(_run("cold", 2.845), _run("parallel-lookups", 1.012))
    assert report["improvement_factor"] == 2.81
    assert report["runs_comparable"] is True
    assert report["all_frames_serialised"] is True
    assert report["cases"] == 132


def test_a_run_pair_over_different_case_counts_reports_no_ratio():
    """The precondition the number never had. 2.81 over 132 vs 96 cases is
    arithmetic, not a finding, and the gate cannot tell from the ratio."""
    report = summarise(_run("cold", 2.845, cases=132),
                       _run("parallel-lookups", 1.012, cases=96))
    assert report["runs_comparable"] is False
    assert report["improvement_factor"] is None
    assert "132 cases" in report["comparability_note"]
    assert "96" in report["comparability_note"]


def test_a_run_whose_frames_did_not_all_serialise_is_flagged():
    report = summarise(_run("cold", 2.845),
                       _run("parallel-lookups", 1.012, frames=130))
    assert report["all_frames_serialised"] is False


def test_no_improvement_is_reported_as_no_improvement():
    """A pipelined run that is not faster must produce a ratio at or below
    1.0, not a rounding that flatters it."""
    report = summarise(_run("cold", 1.000), _run("parallel-lookups", 1.000))
    assert report["improvement_factor"] == 1.0

    slower = summarise(_run("cold", 1.000), _run("parallel-lookups", 2.000))
    assert slower["improvement_factor"] == 0.5


def test_a_zero_wall_time_does_not_divide_by_zero():
    report = summarise(_run("cold", 1.0), _run("parallel-lookups", 0.0))
    assert report["improvement_factor"] > 0


def test_the_raw_runs_survive_into_the_report():
    report = summarise(_run("cold", 2.845), _run("parallel-lookups", 1.012))
    assert [r["run"] for r in report["runs"]] == ["cold", "parallel-lookups"]
    assert report["runs"][0]["stage_p50_ms"]["lookup"] == 20.5


@pytest.mark.parametrize("values,p,expected", [
    ([1.0, 2.0, 3.0, 4.0], 0.5, 3.0),
    ([1.0], 0.95, 1.0),
    ([], 0.5, None),
    ([5.0, 1.0, 3.0], 0.5, 3.0),          # sorts first
])
def test_percentile_handles_the_edges(values, p, expected):
    """Including the empty case: a percentile over no samples is None, not
    0.0 — the gate must be able to tell "fast" from "never measured"."""
    assert percentile(values, p) == expected
