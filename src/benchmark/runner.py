"""Benchmark execution runner and reporting engine for Phase 3B.

This module executes benchmark scenarios, evaluates expectation matches vs.
navigation success, computes quantitative performance metrics, and exports
structured JSON reports to `outputs/metrics/benchmark_results.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

# Ensure workspace root is in sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.scenario import BenchmarkScenario
from src.benchmark.suite import get_default_scenarios
from src.metrics import compute_metrics
from src.simulator import SimulationStatus

DEFAULT_BENCHMARK_OUTPUT = ROOT_DIR / "outputs" / "metrics" / "benchmark_results.json"


@dataclass
class ScenarioResult:
    """Record of execution outcome and metrics for an individual benchmark scenario.

    Attributes
    ----------
    scenario_name : str
        Unique identifier of the evaluated scenario.
    scenario_description : str
        Summary of the scenario objective.
    expected_outcome : str
        Expected terminal condition (e.g. "SUCCESS", "NO_PATH", "ValueError").
    actual_outcome : str
        Observed terminal condition or exception string.
    expectation_met : bool
        True if the actual outcome matched the expected outcome.
    navigation_success : bool
        True if the robot successfully reached the target goal (`SimulationStatus.SUCCESS`).
    is_expected_failure : bool
        True if expectation was met for a non-navigation-success outcome (e.g. expected NO_PATH).
    is_unexpected_failure : bool
        True if the observed outcome diverged from the scenario expectation.
    simulation_status : Optional[str]
        Terminal status string enum if simulation completed, else None.
    exception_type : Optional[str]
        Exception class name if an exception was raised, else None.
    exception_message : Optional[str]
        Exception text if an exception was raised, else None.
    metrics : Optional[Dict[str, Any]]
        Quantitative metrics dictionary from `src.metrics`, or None.
    error_details : Optional[str]
        Diagnostic message if expectation was not met.
    """

    scenario_name: str
    scenario_description: str
    expected_outcome: str
    actual_outcome: str
    expectation_met: bool
    navigation_success: bool
    is_expected_failure: bool
    is_unexpected_failure: bool
    simulation_status: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    controller_name: str = "HeadingProportionalController"

    def as_dict(self) -> Dict[str, Any]:
        """Convert result to a plain JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated benchmark report across all executed scenarios.

    Attributes
    ----------
    schema_version : str
        Report data format version.
    total_scenarios : int
        Total number of scenarios evaluated.
    passed_scenarios : int
        Number of scenarios where expectation was met.
    failed_scenarios : int
        Number of scenarios where expectation was NOT met (unexpected failures).
    benchmark_pass_rate : float
        Fraction of scenarios meeting expectations: `passed_scenarios / total_scenarios`.
    navigation_success_count : int
        Number of scenarios achieving goal arrival (`SimulationStatus.SUCCESS`).
    navigation_success_rate : float
        Fraction of scenarios reaching goal: `navigation_success_count / total_scenarios`.
    results : List[ScenarioResult]
        Detailed scenario-level outcome records.
    controller_name : str
        Controller used in the benchmark suite run.
    """

    schema_version: str = "1.0.0"
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    benchmark_pass_rate: float = 0.0
    navigation_success_count: int = 0
    navigation_success_rate: float = 0.0
    results: List[ScenarioResult] = field(default_factory=list)
    controller_name: str = "HeadingProportionalController"

    def as_dict(self) -> Dict[str, Any]:
        """Convert suite result to a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "controller_name": self.controller_name,
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "failed_scenarios": self.failed_scenarios,
            "benchmark_pass_rate": round(self.benchmark_pass_rate, 4),
            "navigation_success_count": self.navigation_success_count,
            "navigation_success_rate": round(self.navigation_success_rate, 4),
            "results": [r.as_dict() for r in self.results],
        }


class BenchmarkRunner:
    """Execution runner for evaluating a sequence of navigation benchmark scenarios."""

    def __init__(
        self,
        scenarios: Optional[Sequence[BenchmarkScenario]] = None,
        controller: str = "heading",
        lookahead_distance: Optional[float] = None,
    ) -> None:
        """Initialize the benchmark runner with a collection of scenarios.

        Parameters
        ----------
        scenarios : Optional[Sequence[BenchmarkScenario]], optional
            Scenarios to evaluate. If None, defaults to `get_default_scenarios()`.
        controller : str, optional
            Path-tracking controller type ('heading' or 'pure_pursuit'). Defaults to 'heading'.
        lookahead_distance : Optional[float], optional
            Lookahead distance in meters for Pure Pursuit. Defaults to None.
        """
        if scenarios is None:
            self._scenarios: List[BenchmarkScenario] = list(get_default_scenarios())
        else:
            if not isinstance(scenarios, (list, tuple)):
                raise TypeError(f"Scenarios must be a sequence of BenchmarkScenario, got {type(scenarios).__name__}.")
            for s in scenarios:
                if not isinstance(s, BenchmarkScenario):
                    raise TypeError(f"All elements must be BenchmarkScenario instances, got {type(s).__name__}.")
            self._scenarios = list(scenarios)

        self._controller = str(controller)
        self._lookahead_distance = lookahead_distance

    @property
    def scenarios(self) -> List[BenchmarkScenario]:
        """Return the list of configured scenarios."""
        return list(self._scenarios)

    def run_scenario(self, scenario: BenchmarkScenario) -> ScenarioResult:
        """Execute a single benchmark scenario and classify its outcome.

        Parameters
        ----------
        scenario : BenchmarkScenario
            The scenario specification to execute.

        Returns
        -------
        ScenarioResult
            Evaluation outcome record.
        """
        # Case A: Validation Exception Scenario
        if scenario.expected_exception is not None:
            expected_name = scenario.expected_exception.__name__
            sim = scenario.create_simulator(
                controller=self._controller,
                lookahead_distance=self._lookahead_distance,
            )

            # 1. Test primary start/goal
            try:
                sim.run(start=scenario.start, goal=scenario.goal)
                # If no exception raised, it's an unexpected failure
                return ScenarioResult(
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                    expected_outcome=expected_name,
                    actual_outcome="NO_EXCEPTION_RAISED",
                    expectation_met=False,
                    navigation_success=False,
                    is_expected_failure=False,
                    is_unexpected_failure=True,
                    error_details=f"Expected exception {expected_name} was not raised on primary query.",
                    controller_name=sim.controller_name,
                )
            except Exception as exc:
                if not isinstance(exc, scenario.expected_exception):
                    return ScenarioResult(
                        scenario_name=scenario.name,
                        scenario_description=scenario.description,
                        expected_outcome=expected_name,
                        actual_outcome=f"{type(exc).__name__}: {str(exc)}",
                        expectation_met=False,
                        navigation_success=False,
                        is_expected_failure=False,
                        is_unexpected_failure=True,
                        exception_type=type(exc).__name__,
                        exception_message=str(exc),
                        error_details=f"Raised {type(exc).__name__} instead of expected {expected_name}.",
                        controller_name=sim.controller_name,
                    )
                first_exc = exc

            # 2. Test any additional invalid queries
            for q_start, q_goal in scenario.additional_invalid_queries:
                try:
                    sim.run(start=q_start, goal=q_goal)
                    return ScenarioResult(
                        scenario_name=scenario.name,
                        scenario_description=scenario.description,
                        expected_outcome=expected_name,
                        actual_outcome="NO_EXCEPTION_RAISED",
                        expectation_met=False,
                        navigation_success=False,
                        is_expected_failure=False,
                        is_unexpected_failure=True,
                        error_details=f"Expected exception {expected_name} was not raised on query ({q_start}, {q_goal}).",
                        controller_name=sim.controller_name,
                    )
                except Exception as add_exc:
                    if not isinstance(add_exc, scenario.expected_exception):
                        return ScenarioResult(
                            scenario_name=scenario.name,
                            scenario_description=scenario.description,
                            expected_outcome=expected_name,
                            actual_outcome=f"{type(add_exc).__name__}: {str(add_exc)}",
                            expectation_met=False,
                            navigation_success=False,
                            is_expected_failure=False,
                            is_unexpected_failure=True,
                            exception_type=type(add_exc).__name__,
                            exception_message=str(add_exc),
                            error_details=f"Raised {type(add_exc).__name__} on query ({q_start}, {q_goal}).",
                            controller_name=sim.controller_name,
                        )

            # All invalid queries correctly raised expected_exception
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                expected_outcome=expected_name,
                actual_outcome=f"{type(first_exc).__name__}: {str(first_exc)}",
                expectation_met=True,
                navigation_success=False,
                is_expected_failure=True,
                is_unexpected_failure=False,
                exception_type=type(first_exc).__name__,
                exception_message=str(first_exc),
                controller_name=sim.controller_name,
            )

        # Case B: Standard Simulation Scenario
        expected_status = scenario.expected_status
        assert expected_status is not None  # Guaranteed by BenchmarkScenario validation
        expected_name = expected_status.value
        sim = scenario.create_simulator(
            controller=self._controller,
            lookahead_distance=self._lookahead_distance,
        )

        try:
            sim_result = sim.run(start=scenario.start, goal=scenario.goal)
        except Exception as sim_exc:
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                expected_outcome=expected_name,
                actual_outcome=f"EXCEPTION_{type(sim_exc).__name__}: {str(sim_exc)}",
                expectation_met=False,
                navigation_success=False,
                is_expected_failure=False,
                is_unexpected_failure=True,
                exception_type=type(sim_exc).__name__,
                exception_message=str(sim_exc),
                error_details=f"Unexpected exception during simulation: {str(sim_exc)}",
                controller_name=sim.controller_name,
            )

        actual_status = sim_result.status.value
        expectation_met = (sim_result.status == expected_status)
        nav_success = (sim_result.status == SimulationStatus.SUCCESS)
        is_exp_failure = (not nav_success and expectation_met)
        is_unexp_failure = (not expectation_met)

        # Compute metrics using existing src.metrics API
        metrics_dict = compute_metrics(sim_result).as_dict()

        err_msg = None
        if not expectation_met:
            err_msg = f"Observed status '{actual_status}' does not match expected '{expected_name}'."

        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            expected_outcome=expected_name,
            actual_outcome=actual_status,
            expectation_met=expectation_met,
            navigation_success=nav_success,
            is_expected_failure=is_exp_failure,
            is_unexpected_failure=is_unexp_failure,
            simulation_status=actual_status,
            metrics=metrics_dict,
            error_details=err_msg,
            controller_name=sim.controller_name,
        )

    def run_all(self) -> BenchmarkSuiteResult:
        """Execute all configured scenarios and compile the aggregate benchmark report.

        Returns
        -------
        BenchmarkSuiteResult
            Complete suite report with summary metrics and scenario results.
        """
        results: List[ScenarioResult] = []
        passed_count = 0
        failed_count = 0
        nav_success_count = 0
        active_ctrl_name = "HeadingProportionalController"

        for scenario in self._scenarios:
            res = self.run_scenario(scenario)
            results.append(res)
            active_ctrl_name = res.controller_name
            if res.expectation_met:
                passed_count += 1
            else:
                failed_count += 1
            if res.navigation_success:
                nav_success_count += 1

        total = len(self._scenarios)
        pass_rate = (passed_count / total) if total > 0 else 0.0
        nav_rate = (nav_success_count / total) if total > 0 else 0.0

        return BenchmarkSuiteResult(
            total_scenarios=total,
            passed_scenarios=passed_count,
            failed_scenarios=failed_count,
            benchmark_pass_rate=pass_rate,
            navigation_success_count=nav_success_count,
            navigation_success_rate=nav_rate,
            results=results,
            controller_name=active_ctrl_name,
        )

    def export_json(
        self,
        suite_result: BenchmarkSuiteResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Serialize benchmark suite results to a JSON file.

        Parameters
        ----------
        suite_result : BenchmarkSuiteResult
            The benchmark suite results to export.
        output_path : Optional[Path], optional
            Destination path. Defaults to `outputs/metrics/benchmark_results.json`.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        path = output_path or DEFAULT_BENCHMARK_OUTPUT
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(suite_result.as_dict(), f, indent=2)
        return path


def print_suite_report(report: BenchmarkSuiteResult, json_path: Optional[Path] = None) -> None:
    """Print an attractive summary table of benchmark results to standard output."""
    header = "=" * 80
    sub_header = "-" * 80
    print(header)
    print(f" AI-BASED AUTONOMOUS NAVIGATION SYSTEM -- BENCHMARK SUITE [{report.controller_name}]")
    print(header)
    print(f"Total Scenarios Evaluated : {report.total_scenarios}")
    print(f"Benchmark Pass Count      : {report.passed_scenarios} / {report.total_scenarios} ({report.benchmark_pass_rate * 100:.1f}%)")
    print(f"Navigation Success Count  : {report.navigation_success_count} / {report.total_scenarios} ({report.navigation_success_rate * 100:.1f}%)")
    print(f"Unexpected Failures       : {report.failed_scenarios}")
    print(sub_header)
    print(f"{'Scenario Name':<24} | {'Expected':<12} | {'Actual':<14} | {'Match':<6} | {'Nav Success'}")
    print(sub_header)

    for r in report.results:
        match_str = "PASS" if r.expectation_met else "FAIL"
        nav_str = "YES" if r.navigation_success else "NO"
        # Truncate actual outcome for table formatting if needed
        actual_display = r.actual_outcome
        if len(actual_display) > 14:
            actual_display = actual_display[:11] + "..."
        print(f"{r.scenario_name:<24} | {r.expected_outcome:<12} | {actual_display:<14} | {match_str:<6} | {nav_str}")

    print(sub_header)
    if json_path:
        print(f"Report exported to: {json_path}")
    print(header)


def print_comparison_report(
    heading_report: BenchmarkSuiteResult,
    pp_report: BenchmarkSuiteResult,
    json_path: Optional[Path] = None,
) -> None:
    """Print a side-by-side comparative performance report for both controllers."""
    header = "=" * 95
    sub_header = "-" * 95
    print(header)
    print(" AI-BASED AUTONOMOUS NAVIGATION SYSTEM -- CONTROLLER COMPARISON BENCHMARK")
    print(header)

    # Compute aggregate metrics for successful runs
    def _calc_stats(report: BenchmarkSuiteResult) -> Dict[str, Any]:
        success_runs = [r for r in report.results if r.navigation_success and r.metrics]
        collisions = sum(1 for r in report.results if r.simulation_status == "COLLISION")
        timeouts = sum(1 for r in report.results if r.simulation_status == "TIMEOUT")
        total_steps = sum(r.metrics["steps"] for r in success_runs)
        mean_steps = (total_steps / len(success_runs)) if success_runs else None
        trav_dists = [r.metrics["traveled_distance"] for r in success_runs]
        plan_lens = [r.metrics["planned_path_length"] for r in success_runs if r.metrics["planned_path_length"] is not None]
        ratios = [r.metrics["actual_to_planned_ratio"] for r in success_runs if r.metrics["actual_to_planned_ratio"] is not None]
        errors = [r.metrics["final_goal_error"] for r in success_runs if r.metrics["final_goal_error"] is not None]

        return {
            "total": report.total_scenarios,
            "pass_count": report.passed_scenarios,
            "pass_rate": report.benchmark_pass_rate,
            "nav_count": report.navigation_success_count,
            "nav_rate": report.navigation_success_rate,
            "collisions": collisions,
            "timeouts": timeouts,
            "total_steps": total_steps,
            "mean_steps": mean_steps,
            "mean_dist": (sum(trav_dists) / len(trav_dists)) if trav_dists else None,
            "mean_plan": (sum(plan_lens) / len(plan_lens)) if plan_lens else None,
            "mean_ratio": (sum(ratios) / len(ratios)) if ratios else None,
            "mean_error": (sum(errors) / len(errors)) if errors else None,
        }

    h_stats = _calc_stats(heading_report)
    p_stats = _calc_stats(pp_report)

    print(f"{'Aggregate Performance Metric':<35} | {'Heading Proportional':<25} | {'Pure Pursuit':<25}")
    print(sub_header)
    print(f"{'Benchmark Pass Rate':<35} | {h_stats['pass_count']}/{h_stats['total']} ({h_stats['pass_rate']*100:.1f}%)"
          f"{'':<13} | {p_stats['pass_count']}/{p_stats['total']} ({p_stats['pass_rate']*100:.1f}%)")
    print(f"{'Navigation Success Rate':<35} | {h_stats['nav_count']}/{h_stats['total']} ({h_stats['nav_rate']*100:.1f}%)"
          f"{'':<13} | {p_stats['nav_count']}/{p_stats['total']} ({p_stats['nav_rate']*100:.1f}%)")
    print(f"{'Collision Count':<35} | {h_stats['collisions']:<25} | {p_stats['collisions']:<25}")
    print(f"{'Timeout Count':<35} | {h_stats['timeouts']:<25} | {p_stats['timeouts']:<25}")
    print(f"{'Total Steps (Successful)':<35} | {h_stats['total_steps']:<25} | {p_stats['total_steps']:<25}")
    h_ms = f"{h_stats['mean_steps']:.1f}" if h_stats['mean_steps'] is not None else "N/A"
    p_ms = f"{p_stats['mean_steps']:.1f}" if p_stats['mean_steps'] is not None else "N/A"
    print(f"{'Mean Steps per Success':<35} | {h_ms:<25} | {p_ms:<25}")
    h_md = f"{h_stats['mean_dist']:.3f} m" if h_stats['mean_dist'] is not None else "N/A"
    p_md = f"{p_stats['mean_dist']:.3f} m" if p_stats['mean_dist'] is not None else "N/A"
    print(f"{'Mean Traveled Distance':<35} | {h_md:<25} | {p_md:<25}")
    h_mp = f"{h_stats['mean_plan']:.3f} m" if h_stats['mean_plan'] is not None else "N/A"
    p_mp = f"{p_stats['mean_plan']:.3f} m" if p_stats['mean_plan'] is not None else "N/A"
    print(f"{'Mean Planned Length':<35} | {h_mp:<25} | {p_mp:<25}")
    h_mr = f"{h_stats['mean_ratio']:.3f}" if h_stats['mean_ratio'] is not None else "N/A"
    p_mr = f"{p_stats['mean_ratio']:.3f}" if p_stats['mean_ratio'] is not None else "N/A"
    print(f"{'Mean Actual/Planned Ratio':<35} | {h_mr:<25} | {p_mr:<25}")
    h_me = f"{h_stats['mean_error']:.3f} m" if h_stats['mean_error'] is not None else "N/A"
    p_me = f"{p_stats['mean_error']:.3f} m" if p_stats['mean_error'] is not None else "N/A"
    print(f"{'Mean Final Goal Error':<35} | {h_me:<25} | {p_me:<25}")
    print(sub_header)

    print(f"{'Scenario':<20} | {'Controller':<15} | {'Outcome':<10} | {'Steps':<6} | {'Plan (m)':<8} | {'Trav (m)':<8} | {'Ratio':<6} | {'Error (m)'}")
    print(sub_header)

    pp_dict = {r.scenario_name: r for r in pp_report.results}
    for hr in heading_report.results:
        pr = pp_dict.get(hr.scenario_name)

        def _fmt_row(name: str, ctrl: str, r: Optional[ScenarioResult]) -> str:
            if r is None:
                return f"{name:<20} | {ctrl:<15} | {'MISSING':<10} | {'-':<6} | {'-':<8} | {'-':<8} | {'-':<6} | {'-'}"
            st = r.actual_outcome
            if len(st) > 10:
                st = st[:8] + ".."
            if r.metrics:
                steps_str = str(r.metrics["steps"])
                plan_str = f"{r.metrics['planned_path_length']:.2f}" if r.metrics["planned_path_length"] is not None else "N/A"
                trav_str = f"{r.metrics['traveled_distance']:.2f}"
                ratio_str = f"{r.metrics['actual_to_planned_ratio']:.3f}" if r.metrics["actual_to_planned_ratio"] is not None else "N/A"
                err_str = f"{r.metrics['final_goal_error']:.3f}" if r.metrics["final_goal_error"] is not None else "N/A"
            else:
                steps_str, plan_str, trav_str, ratio_str, err_str = "-", "-", "-", "-", "-"
            return f"{name:<20} | {ctrl:<15} | {st:<10} | {steps_str:<6} | {plan_str:<8} | {trav_str:<8} | {ratio_str:<6} | {err_str}"

        print(_fmt_row(hr.scenario_name, "Heading", hr))
        print(_fmt_row("", "PurePursuit", pr))
        print("-" * 95)

    if json_path:
        print(f"Comparison report exported to: {json_path}")
    print(header)


def main() -> int:
    """CLI entrypoint for executing the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Run deterministic navigation benchmark suite (Phase 3B / Pure Pursuit)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_BENCHMARK_OUTPUT),
        help=f"Path to export benchmark results JSON. Default: {DEFAULT_BENCHMARK_OUTPUT}",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="heading",
        choices=["heading", "pure_pursuit"],
        help="Controller algorithm to benchmark ('heading' or 'pure_pursuit', default: 'heading').",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=None,
        help="Lookahead distance in meters for pure pursuit controller (default: 0.5).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both Heading and Pure Pursuit controllers and print a comparative report.",
    )
    parser.add_argument(
        "--tune-pure-pursuit",
        action="store_true",
        help="Run systematic lookahead tuning experiment across distances [0.25, 0.5, 0.75, 1.0] m.",
    )
    args = parser.parse_args()

    if args.tune_pure_pursuit:
        from src.benchmark.tuning import run_pure_pursuit_tuning, print_tuning_report, DEFAULT_TUNING_OUTPUT
        tuning_out = Path(args.output) if args.output != str(DEFAULT_BENCHMARK_OUTPUT) else DEFAULT_TUNING_OUTPUT
        tuning_report = run_pure_pursuit_tuning(output_path=tuning_out)
        print_tuning_report(tuning_report, json_path=tuning_out)
        return 0

    if args.compare:
        heading_runner = BenchmarkRunner(controller="heading")
        heading_report = heading_runner.run_all()
        heading_runner.export_json(
            heading_report,
            output_path=Path("outputs/metrics/benchmark_results_heading.json"),
        )

        pp_runner = BenchmarkRunner(controller="pure_pursuit", lookahead_distance=args.lookahead)
        pp_report = pp_runner.run_all()
        pp_runner.export_json(
            pp_report,
            output_path=Path("outputs/metrics/benchmark_results_pure_pursuit.json"),
        )

        comp_json_path = Path("outputs/metrics/benchmark_comparison.json")
        comp_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": "1.0.0",
                    "heading_controller": heading_report.as_dict(),
                    "pure_pursuit_controller": pp_report.as_dict(),
                },
                f,
                indent=2,
            )

        print_comparison_report(heading_report, pp_report, json_path=comp_json_path)
        return 0 if (heading_report.failed_scenarios == 0 and pp_report.failed_scenarios == 0) else 1

    runner = BenchmarkRunner(controller=args.controller, lookahead_distance=args.lookahead)
    report = runner.run_all()
    exported_path = runner.export_json(report, output_path=Path(args.output))
    print_suite_report(report, json_path=exported_path)

    # Return exit code 0 if all expectations were met, else 1
    return 0 if report.failed_scenarios == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
