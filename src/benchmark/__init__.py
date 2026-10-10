"""Benchmark evaluation suite package for the AI-Based Autonomous Navigation System.

This package provides deterministic, reproducible benchmark evaluation scenarios
and reporting mechanisms for mobile robot navigation (Phase 3B).
"""

from src.benchmark.scenario import BenchmarkScenario
from src.benchmark.suite import (
    get_default_scenarios,
    get_scenario_by_name,
    scenario_invalid_start_goal,
    scenario_maze,
    scenario_multiple_obstacles,
    scenario_narrow_corridor,
    scenario_one_obstacle,
    scenario_open_space,
    scenario_unreachable_goal,
)


def __getattr__(name: str):
    if name in ("BenchmarkRunner", "BenchmarkSuiteResult", "ScenarioResult"):
        import src.benchmark.runner as runner_mod

        return getattr(runner_mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BenchmarkScenario",
    "ScenarioResult",
    "BenchmarkSuiteResult",
    "BenchmarkRunner",
    "get_default_scenarios",
    "get_scenario_by_name",
    "scenario_open_space",
    "scenario_one_obstacle",
    "scenario_multiple_obstacles",
    "scenario_narrow_corridor",
    "scenario_maze",
    "scenario_unreachable_goal",
    "scenario_invalid_start_goal",
]
