"""Systematic lookahead distance tuning and robustness experiment for Pure Pursuit.

This module evaluates the unicycle Pure Pursuit controller across a spectrum
of lookahead distances (0.25m, 0.5m, 0.75m, 1.0m) across all seven standardized
benchmark scenarios, comparing them against the baseline Heading Proportional controller.

Outputs a structured, versioned JSON report to:
`outputs/metrics/pure_pursuit_tuning.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.runner import BenchmarkRunner, BenchmarkSuiteResult, ScenarioResult
from src.benchmark.scenario import BenchmarkScenario
from src.benchmark.suite import get_default_scenarios
from src.simulator import SimulationStatus

DEFAULT_TUNING_OUTPUT = ROOT_DIR / "outputs" / "metrics" / "pure_pursuit_tuning.json"
DEFAULT_LOOKAHEAD_CANDIDATES: Sequence[float] = (0.25, 0.5, 0.75, 1.0)


@dataclass
class ScenarioRunRecord:
    """Detailed record of a single scenario execution within the tuning experiment."""

    scenario_name: str
    scenario_description: str
    controller: str
    lookahead_distance: Optional[float]
    expected_outcome: str
    terminal_status: str
    expectation_met: bool
    navigation_success: bool
    is_navigable_scenario: bool
    is_collision: bool
    is_timeout: bool
    steps: Optional[int]
    planned_path_length: Optional[float]
    traveled_distance: Optional[float]
    actual_to_planned_ratio: Optional[float]
    final_goal_error: Optional[float]
    error_details: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class LookaheadAggregateMetrics:
    """Quantitative summary metrics for an evaluated lookahead distance."""

    controller: str
    lookahead_distance: Optional[float]
    total_scenarios_evaluated: int
    navigable_scenarios_count: int
    navigation_successes: int
    navigation_success_rate: float
    collisions: int
    timeouts: int
    mean_steps_successful: Optional[float]
    mean_traveled_distance_successful: Optional[float]
    mean_actual_to_planned_ratio_successful: Optional[float]
    mean_final_goal_error_successful: Optional[float]
    expected_non_navigable_matches: int
    total_benchmark_matches: int
    benchmark_pass_rate: float

    def as_dict(self) -> Dict[str, Any]:
        """Convert summary to a JSON-serializable dictionary."""
        d = asdict(self)
        # Round floating points cleanly
        for key in (
            "navigation_success_rate",
            "mean_steps_successful",
            "mean_traveled_distance_successful",
            "mean_actual_to_planned_ratio_successful",
            "mean_final_goal_error_successful",
            "benchmark_pass_rate",
        ):
            if d[key] is not None:
                d[key] = round(d[key], 4)
        return d


@dataclass
class LookaheadExperimentSet:
    """Complete collection of scenario runs and aggregates for a specific configuration."""

    configuration_label: str
    controller: str
    lookahead_distance: Optional[float]
    aggregate_metrics: LookaheadAggregateMetrics
    scenario_runs: List[ScenarioRunRecord] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "configuration_label": self.configuration_label,
            "controller": self.controller,
            "lookahead_distance": self.lookahead_distance,
            "aggregate_metrics": self.aggregate_metrics.as_dict(),
            "scenario_runs": [r.as_dict() for r in self.scenario_runs],
        }


@dataclass
class PurePursuitTuningReport:
    """Root container for lookahead tuning experiment outcomes."""

    schema_version: str = "1.0.0"
    experiment_name: str = "pure_pursuit_lookahead_tuning"
    evaluated_lookaheads: List[float] = field(default_factory=list)
    baseline_controller: str = "HeadingProportionalController"
    baseline_experiment: Optional[LookaheadExperimentSet] = None
    lookahead_experiments: Dict[str, LookaheadExperimentSet] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert full report to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "evaluated_lookaheads": self.evaluated_lookaheads,
            "baseline_controller": self.baseline_controller,
            "baseline_experiment": (
                self.baseline_experiment.as_dict() if self.baseline_experiment else None
            ),
            "lookahead_experiments": {
                k: v.as_dict() for k, v in self.lookahead_experiments.items()
            },
        }


def _evaluate_configuration(
    scenarios: Sequence[BenchmarkScenario],
    controller: str,
    lookahead_distance: Optional[float],
) -> LookaheadExperimentSet:
    """Run all scenarios under a specified controller and lookahead configuration."""
    runner = BenchmarkRunner(
        scenarios=scenarios,
        controller=controller,
        lookahead_distance=lookahead_distance,
    )
    suite_res: BenchmarkSuiteResult = runner.run_all()

    runs: List[ScenarioRunRecord] = []
    navigable_count = 0
    nav_success_count = 0
    collision_count = 0
    timeout_count = 0
    expected_non_nav_matches = 0

    success_steps: List[int] = []
    success_dists: List[float] = []
    success_ratios: List[float] = []
    success_errors: List[float] = []

    for sr in suite_res.results:
        # Determine if scenario was designed to be navigable
        is_navigable = (sr.expected_outcome == SimulationStatus.SUCCESS.value)
        if is_navigable:
            navigable_count += 1
            if sr.navigation_success:
                nav_success_count += 1
        else:
            if sr.expectation_met:
                expected_non_nav_matches += 1

        is_coll = (sr.simulation_status == SimulationStatus.COLLISION.value)
        is_to = (sr.simulation_status == SimulationStatus.TIMEOUT.value)
        if is_coll:
            collision_count += 1
        if is_to:
            timeout_count += 1

        # Extract metric details if available
        steps_val = None
        plan_val = None
        dist_val = None
        ratio_val = None
        err_val = None

        if sr.metrics:
            steps_val = sr.metrics.get("steps")
            plan_val = sr.metrics.get("planned_path_length")
            dist_val = sr.metrics.get("traveled_distance")
            ratio_val = sr.metrics.get("actual_to_planned_ratio")
            err_val = sr.metrics.get("final_goal_error")

            if sr.navigation_success and is_navigable:
                if steps_val is not None:
                    success_steps.append(steps_val)
                if dist_val is not None:
                    success_dists.append(dist_val)
                if ratio_val is not None:
                    success_ratios.append(ratio_val)
                if err_val is not None:
                    success_errors.append(err_val)

        record = ScenarioRunRecord(
            scenario_name=sr.scenario_name,
            scenario_description=sr.scenario_description,
            controller=sr.controller_name,
            lookahead_distance=lookahead_distance,
            expected_outcome=sr.expected_outcome,
            terminal_status=sr.actual_outcome,
            expectation_met=sr.expectation_met,
            navigation_success=sr.navigation_success,
            is_navigable_scenario=is_navigable,
            is_collision=is_coll,
            is_timeout=is_to,
            steps=steps_val,
            planned_path_length=plan_val,
            traveled_distance=dist_val,
            actual_to_planned_ratio=ratio_val,
            final_goal_error=err_val,
            error_details=sr.error_details,
        )
        runs.append(record)

    total_scenarios = len(scenarios)
    nav_rate = (nav_success_count / navigable_count) if navigable_count > 0 else 0.0
    pass_rate = (suite_res.passed_scenarios / total_scenarios) if total_scenarios > 0 else 0.0

    mean_steps = (sum(success_steps) / len(success_steps)) if success_steps else None
    mean_dist = (sum(success_dists) / len(success_dists)) if success_dists else None
    mean_ratio = (sum(success_ratios) / len(success_ratios)) if success_ratios else None
    mean_err = (sum(success_errors) / len(success_errors)) if success_errors else None

    ctrl_label = suite_res.controller_name
    if lookahead_distance is not None:
        cfg_label = f"PurePursuit (L_d={lookahead_distance:.2f}m)"
    else:
        cfg_label = f"HeadingProportional (Baseline)"

    aggregates = LookaheadAggregateMetrics(
        controller=ctrl_label,
        lookahead_distance=lookahead_distance,
        total_scenarios_evaluated=total_scenarios,
        navigable_scenarios_count=navigable_count,
        navigation_successes=nav_success_count,
        navigation_success_rate=nav_rate,
        collisions=collision_count,
        timeouts=timeout_count,
        mean_steps_successful=mean_steps,
        mean_traveled_distance_successful=mean_dist,
        mean_actual_to_planned_ratio_successful=mean_ratio,
        mean_final_goal_error_successful=mean_err,
        expected_non_navigable_matches=expected_non_nav_matches,
        total_benchmark_matches=suite_res.passed_scenarios,
        benchmark_pass_rate=pass_rate,
    )

    return LookaheadExperimentSet(
        configuration_label=cfg_label,
        controller=ctrl_label,
        lookahead_distance=lookahead_distance,
        aggregate_metrics=aggregates,
        scenario_runs=runs,
    )


def run_pure_pursuit_tuning(
    lookaheads: Sequence[float] = DEFAULT_LOOKAHEAD_CANDIDATES,
    output_path: Optional[Path] = None,
) -> PurePursuitTuningReport:
    """Execute systematic lookahead distance evaluation across benchmark scenarios.

    Parameters
    ----------
    lookaheads : Sequence[float], optional
        List of lookahead distances in meters to evaluate. Defaults to [0.25, 0.5, 0.75, 1.0].
    output_path : Optional[Path], optional
        Destination path for JSON results. Defaults to outputs/metrics/pure_pursuit_tuning.json.

    Returns
    -------
    PurePursuitTuningReport
        Complete tuning report object.
    """
    scenarios = list(get_default_scenarios())

    # 1. Run baseline Heading Controller
    baseline_set = _evaluate_configuration(
        scenarios=scenarios,
        controller="heading",
        lookahead_distance=None,
    )

    # 2. Run Pure Pursuit across all candidate lookaheads
    lookahead_experiments: Dict[str, LookaheadExperimentSet] = {}
    for ld in lookaheads:
        exp_set = _evaluate_configuration(
            scenarios=scenarios,
            controller="pure_pursuit",
            lookahead_distance=float(ld),
        )
        lookahead_experiments[str(ld)] = exp_set

    report = PurePursuitTuningReport(
        schema_version="1.0.0",
        experiment_name="pure_pursuit_lookahead_tuning",
        evaluated_lookaheads=[float(x) for x in lookaheads],
        baseline_controller="HeadingProportionalController",
        baseline_experiment=baseline_set,
        lookahead_experiments=lookahead_experiments,
    )

    # Export to disk
    dest = output_path or DEFAULT_TUNING_OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, indent=2)

    return report


def print_tuning_report(report: PurePursuitTuningReport, json_path: Optional[Path] = None) -> None:
    """Render a comprehensive formatted report to standard output."""
    header = "=" * 105
    sub_header = "-" * 105
    print(header)
    print(" AI-BASED AUTONOMOUS NAVIGATION SYSTEM -- PURE PURSUIT LOOKAHEAD TUNING EXPERIMENT")
    print(header)

    # Table 1: Aggregate Summary
    print(f"{'Configuration':<30} | {'Nav Success':<14} | {'Collisions':<10} | {'Timeouts':<9} | {'Mean Steps':<11} | {'Mean Dist':<10} | {'Mean Error'}")
    print(sub_header)

    if report.baseline_experiment:
        b_agg = report.baseline_experiment.aggregate_metrics
        b_steps = f"{b_agg.mean_steps_successful:.1f}" if b_agg.mean_steps_successful is not None else "N/A"
        b_dist = f"{b_agg.mean_traveled_distance_successful:.2f} m" if b_agg.mean_traveled_distance_successful is not None else "N/A"
        b_err = f"{b_agg.mean_final_goal_error_successful:.3f} m" if b_agg.mean_final_goal_error_successful is not None else "N/A"
        print(f"{b_agg.controller + ' (Base)':<30} | {b_agg.navigation_successes}/{b_agg.navigable_scenarios_count} ({b_agg.navigation_success_rate*100:.0f}%)"
              f"{'':<4} | {b_agg.collisions:<10} | {b_agg.timeouts:<9} | {b_steps:<11} | {b_dist:<10} | {b_err}")

    for ld_str, exp in report.lookahead_experiments.items():
        agg = exp.aggregate_metrics
        label = f"PurePursuit (L_d={float(ld_str):.2f}m)"
        m_steps = f"{agg.mean_steps_successful:.1f}" if agg.mean_steps_successful is not None else "N/A"
        m_dist = f"{agg.mean_traveled_distance_successful:.2f} m" if agg.mean_traveled_distance_successful is not None else "N/A"
        m_err = f"{agg.mean_final_goal_error_successful:.3f} m" if agg.mean_final_goal_error_successful is not None else "N/A"
        print(f"{label:<30} | {agg.navigation_successes}/{agg.navigable_scenarios_count} ({agg.navigation_success_rate*100:.0f}%)"
              f"{'':<4} | {agg.collisions:<10} | {agg.timeouts:<9} | {m_steps:<11} | {m_dist:<10} | {m_err}")

    print(sub_header)
    print("Detailed Scenario-by-Scenario Matrix (Terminal Outcome / Steps):")
    print(sub_header)

    scenario_names = [r.scenario_name for r in list(report.lookahead_experiments.values())[0].scenario_runs]
    col_w = 16
    print(f"{'Scenario':<22} | {'Heading':<{col_w}}", end="")
    for ld in report.evaluated_lookaheads:
        print(f" | {f'PP (L_d={ld})':<{col_w}}", end="")
    print()
    print(sub_header)

    for s_name in scenario_names:
        # Baseline outcome
        b_run = next((r for r in report.baseline_experiment.scenario_runs if r.scenario_name == s_name), None) if report.baseline_experiment else None
        b_str = f"{b_run.terminal_status} ({b_run.steps}s)" if (b_run and b_run.steps is not None) else (b_run.terminal_status[:col_w] if b_run else "-")
        print(f"{s_name:<22} | {b_str:<{col_w}}", end="")

        for ld in report.evaluated_lookaheads:
            exp = report.lookahead_experiments.get(str(ld))
            s_run = next((r for r in exp.scenario_runs if r.scenario_name == s_name), None) if exp else None
            if s_run and s_run.steps is not None:
                cell_str = f"{s_run.terminal_status} ({s_run.steps}s)"
            elif s_run:
                cell_str = s_run.terminal_status[:col_w]
            else:
                cell_str = "-"
            print(f" | {cell_str:<{col_w}}", end="")
        print()

    print(sub_header)
    if json_path:
        print(f"Tuning experiment report exported to: {json_path}")
    print(header)


def main() -> int:
    """CLI entrypoint for executing lookahead tuning directly."""
    parser = argparse.ArgumentParser(
        description="Run Pure Pursuit lookahead distance tuning experiment."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TUNING_OUTPUT,
        help=f"Destination path for tuning JSON report. Default: {DEFAULT_TUNING_OUTPUT}",
    )
    parser.add_argument(
        "--lookaheads",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Lookahead distances in meters to evaluate (default: 0.25 0.5 0.75 1.0).",
    )
    args = parser.parse_args()

    report = run_pure_pursuit_tuning(lookaheads=args.lookaheads, output_path=args.output)
    print_tuning_report(report, json_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
