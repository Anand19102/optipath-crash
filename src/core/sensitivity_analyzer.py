from typing import Dict, List, Any
import pandas as pd
from src.core.models import ProjectSchedule
from src.core.lp_crashing_solver import LPCrashingSolver


class SensitivityAnalyzer:
    """
    Automated Sensitivity and What-If Analysis engine for project schedules.
    Performs deadline sweeps, computes cost frontiers, and tracks path criticality dynamics.
    """
    def __init__(self, schedule: ProjectSchedule):
        self.schedule = schedule
        self.solver = LPCrashingSolver(schedule)
        self.normal_duration = self.solver.normal_project_duration
        self.min_duration = self.solver.min_project_duration

    def run_deadline_sweep(self, step_size: float = 1.0) -> Dict[str, Any]:
        """
        Sweeps target deadlines from normal duration down to absolute minimum crashable duration.
        Computes direct, indirect, and combined costs at every step to locate the economic minimum.
        """
        current_deadline = float(self.normal_duration)
        sweep_results = []
        
        while current_deadline >= self.min_duration - 1e-5:
            res = self.solver.solve_crashing(target_deadline=round(current_deadline, 2))
            
            if res["status"] == "Optimal":
                crashed_tasks_summary = {
                    tid: d["crashed_amount"] 
                    for tid, d in res["crash_schedule"].items() 
                    if d["crashed_amount"] > 0
                }
                
                sweep_results.append({
                    "target_deadline": round(current_deadline, 2),
                    "achieved_duration": res["achieved_duration"],
                    "time_saved": res["total_time_saved"],
                    "direct_crashing_cost": res["total_crashing_cost"],
                    "indirect_cost": res["total_indirect_cost"],
                    "combined_cost": res["combined_cost"],
                    "marginal_crash_cost (shadow_price)": res["deadline_shadow_price"],
                    "critical_tasks": res["critical_tasks"],
                    "crashed_tasks": crashed_tasks_summary
                })
            
            current_deadline -= step_size

        df = pd.DataFrame(sweep_results)
        
        # Identify the Economic Optimal Duration (Lowest Combined Cost)
        optimal_row = df.loc[df["combined_cost"].idxmin()] if not df.empty else None

        return {
            "sweep_table": df,
            "optimal_duration_point": optimal_row.to_dict() if optimal_row is not None else {},
            "normal_duration": self.normal_duration,
            "min_duration": self.min_duration
        }

    def evaluate_task_delay_impact(self, task_id: str, delay_amount: float) -> Dict[str, Any]:
        """
        What-If Analysis: Evaluates the cascading consequence if an individual task slips.
        Checks if the task delay exceeds its float and expands the project timeline.
        """
        baseline_cpm = self.solver.baseline_cpm
        if task_id not in baseline_cpm["total_float"]:
            raise ValueError(f"Task '{task_id}' not found in project schedule.")

        task_tf = baseline_cpm["total_float"][task_id]
        is_originally_critical = task_id in baseline_cpm["critical_tasks"]
        
        # New duration with delay
        modified_durations = {
            t.id: (t.normal_time + delay_amount if t.id == task_id else t.normal_time)
            for t in self.schedule.tasks
        }
        
        new_cpm = self.solver.cpm_engine.compute_schedule(custom_durations=modified_durations)
        duration_delta = new_cpm["total_duration"] - self.normal_duration

        return {
            "task_id": task_id,
            "delay_injected": delay_amount,
            "original_total_float": task_tf,
            "was_critical_initially": is_originally_critical,
            "new_project_duration": new_cpm["total_duration"],
            "project_delay_incurred": duration_delta,
            "absorbed_by_slack": (duration_delta == 0),
            "new_critical_tasks": new_cpm["critical_tasks"]
        }