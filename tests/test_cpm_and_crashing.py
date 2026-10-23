import json
import pytest
from pydantic import ValidationError
from src.core.models import ProjectSchedule, Task
from src.core.cpm_engine import CPMEngine
from src.core.lp_crashing_solver import LPCrashingSolver
from src.core.sensitivity_analyzer import SensitivityAnalyzer


@pytest.fixture
def kaushik_schedule() -> ProjectSchedule:
    with open("data/kaushik_mills.json") as f:
        data = json.load(f)
    return ProjectSchedule(**data)


@pytest.fixture
def cloud_schedule() -> ProjectSchedule:
    with open("data/cloud_migration.json") as f:
        data = json.load(f)
    return ProjectSchedule(**data)


# ==========================================
# 1. Validation & Graph Error Handling Tests
# ==========================================

def test_invalid_task_durations():
    """Assert error if crash time exceeds normal time."""
    with pytest.raises(ValidationError):
        Task(
            id="T_ERR",
            description="Bad Task",
            normal_time=5.0,
            crash_time=8.0,  # Invalid: crash > normal
            crash_cost_per_unit=100.0
        )


def test_cyclic_dependency_detection():
    """Assert CPMEngine flags circular task dependencies."""
    cyclic_tasks = [
        Task(id="X", normal_time=3, crash_time=3, predecessors=["Z"]),
        Task(id="Y", normal_time=4, crash_time=4, predecessors=["X"]),
        Task(id="Z", normal_time=2, crash_time=2, predecessors=["Y"]),
    ]
    cyclic_schedule = ProjectSchedule(project_name="CycleTest", tasks=cyclic_tasks)
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        CPMEngine(cyclic_schedule)


def test_missing_predecessor_reference():
    """Assert CPMEngine flags non-existent predecessors."""
    broken_tasks = [
        Task(id="T1", normal_time=5, crash_time=5, predecessors=["NON_EXISTENT"])
    ]
    broken_schedule = ProjectSchedule(project_name="BrokenTest", tasks=broken_tasks)
    with pytest.raises(ValueError, match="does not exist"):
        CPMEngine(broken_schedule)


# ==========================================
# 2. Kaushik Mills Ground Truth CPM Tests
# ==========================================

def test_kaushik_baseline_duration(kaushik_schedule):
    """Verify normal baseline project duration is exactly 50.0 weeks."""
    engine = CPMEngine(kaushik_schedule)
    res = engine.compute_schedule()
    assert res["total_duration"] == 50.0


def test_kaushik_critical_path(kaushik_schedule):
    """Verify the identified critical path matches theoretical calculation."""
    engine = CPMEngine(kaushik_schedule)
    res = engine.compute_schedule()
    expected_critical_tasks = ["A", "C", "F", "H", "J", "K"]
    assert set(res["critical_tasks"]) == set(expected_critical_tasks)
    assert [["A", "C", "F", "H", "J", "K"]] == res["critical_paths"]


def test_kaushik_floats(kaushik_schedule):
    """Verify slack/float calculations on non-critical tasks."""
    engine = CPMEngine(kaushik_schedule)
    res = engine.compute_schedule()
    # Task B total float = 50 - 46 = 4 weeks
    assert res["total_float"]["B"] == 4.0
    # Task D total float = 50 - 45 = 5 weeks
    assert res["total_float"]["D"] == 5.0
    # Critical tasks have 0 float
    assert res["total_float"]["A"] == 0.0
    assert res["total_float"]["H"] == 0.0


# ==========================================
# 3. LP Solver Ground Truth Crashing Tests
# ==========================================

def test_lp_crashing_to_48_weeks(kaushik_schedule):
    """Verify crashing from 50 to 48 weeks compresses Task C only at Rs 6,000."""
    solver = LPCrashingSolver(kaushik_schedule)
    res = solver.solve_crashing(target_deadline=48.0)

    assert res["status"] == "Optimal"
    assert res["total_crashing_cost"] == 6000.0
    assert res["deadline_shadow_price"] == 3000.0
    assert res["crash_schedule"]["C"]["crashed_amount"] == 2.0
    # All other tasks remain uncrashed
    for t_id, data in res["crash_schedule"].items():
        if t_id != "C":
            assert data["crashed_amount"] == 0.0


def test_lp_crashing_to_46_weeks(kaushik_schedule):
    """Verify crashing to 46 weeks compresses C (2 wks) and A (2 wks) at Rs 54,000."""
    solver = LPCrashingSolver(kaushik_schedule)
    res = solver.solve_crashing(target_deadline=46.0)

    assert res["status"] == "Optimal"
    assert res["total_crashing_cost"] == 54000.0
    assert res["crash_schedule"]["C"]["crashed_amount"] == 2.0
    assert res["crash_schedule"]["A"]["crashed_amount"] == 2.0
    assert res["deadline_shadow_price"] == 24000.0


def test_lp_crashing_infeasible_limit(kaushik_schedule):
    """Verify attempting to crash past 43.0 weeks returns Infeasible status."""
    solver = LPCrashingSolver(kaushik_schedule)
    res = solver.solve_crashing(target_deadline=40.0)
    assert res["status"] == "Infeasible"
    assert res["min_possible_duration"] == 43.0


# ==========================================
# 4. Sensitivity & Economic Optimum Tests
# ==========================================

def test_economic_optimal_duration(kaushik_schedule):
    """Verify the lowest combined cost occurs at exactly 44.0 weeks."""
    analyzer = SensitivityAnalyzer(kaushik_schedule)
    sweep_res = analyzer.run_deadline_sweep(step_size=1.0)
    opt = sweep_res["optimal_duration_point"]

    assert opt["target_deadline"] == 44.0
    assert opt["combined_cost"] == 3194000.0
    assert opt["direct_crashing_cost"] == 114000.0


def test_what_if_slack_absorption(kaushik_schedule):
    """Verify delay within float is absorbed without shifting project completion."""
    analyzer = SensitivityAnalyzer(kaushik_schedule)
    delay_res = analyzer.evaluate_task_delay_impact(task_id="B", delay_amount=3.0)
    
    assert delay_res["project_delay_incurred"] == 0.0
    assert delay_res["absorbed_by_slack"] is True