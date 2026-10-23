from typing import Dict, List, Optional, Any
import pulp
from src.core.models import ProjectSchedule, Task
from src.core.cpm_engine import CPMEngine


class LPCrashingSolver:
    """
    Linear Programming Optimizer for Project Crashing using PuLP.
    Solves for the exact minimal crashing expenditure to hit any given project deadline.
    """
    def __init__(self, schedule: ProjectSchedule):
        self.schedule = schedule
        self.tasks: Dict[str, Task] = {t.id: t for t in schedule.tasks}
        self.cpm_engine = CPMEngine(schedule)
        
        # Calculate normal (uncompressed) baseline
        self.baseline_cpm = self.cpm_engine.compute_schedule()
        self.normal_project_duration = self.baseline_cpm["total_duration"]
        
        # Calculate minimum possible duration (all tasks fully crashed)
        all_crashed_durations = {t.id: t.crash_time for t in self.schedule.tasks}
        self.min_project_duration = self.cpm_engine.compute_schedule(
            custom_durations=all_crashed_durations
        )["total_duration"]

    def solve_crashing(self, target_deadline: float) -> Dict[str, Any]:
        """
        Formulates and solves the Continuous Linear Program for a specific target deadline.
        
        Parameters:
            target_deadline: Desired project completion duration (in time units).
        
        Returns:
            Dictionary containing optimal crash amounts, costs, updated schedule, and dual values.
        """
        # Safety bound check
        if target_deadline < self.min_project_duration - 1e-5:
            return {
                "status": "Infeasible",
                "message": (
                    f"Target deadline {target_deadline} {self.schedule.time_unit} is unachievable. "
                    f"Absolute physical crash limit of the project is {self.min_project_duration} {self.schedule.time_unit}."
                ),
                "target_deadline": target_deadline,
                "min_possible_duration": self.min_project_duration
            }

        # Initialize LP Minimization Problem
        prob = pulp.LpProblem("Project_Crashing_Optimization", pulp.LpMinimize)

        # Decision Variables
        # T[i]: Start time of task i
        T = {
            t_id: pulp.LpVariable(f"T_{t_id}", lowBound=0, cat=pulp.LpContinuous)
            for t_id in self.tasks
        }
        
        # y[i]: Time duration crashed for task i
        y = {
            t_id: pulp.LpVariable(
                f"y_{t_id}", 
                lowBound=0, 
                upBound=task.max_crash, 
                cat=pulp.LpContinuous
            )
            for t_id, task in self.tasks.items()
        }
        
        # Overall completion time variable
        T_finish = pulp.LpVariable("T_finish", lowBound=0, cat=pulp.LpContinuous)

        # Objective Function: Minimize Total Crashing Cost sum(K_i * y_i)
        prob += (
            pulp.lpSum([self.tasks[t_id].crash_cost_per_unit * y[t_id] for t_id in self.tasks]),
            "Total_Crashing_Cost"
        )

        # Precedence Constraints: T[node] >= T[pred] + (D_pred - y[pred])
        for t_id, task in self.tasks.items():
            for pred in task.predecessors:
                D_pred = self.tasks[pred].normal_time
                prob += (
                    T[t_id] - T[pred] + y[pred] >= D_pred,
                    f"Precedence_{pred}_to_{t_id}"
                )

        # Terminal Constraints: T_finish >= T[end_task] + (D_end - y[end_task])
        end_tasks = [
            t_id for t_id in self.tasks 
            if self.cpm_engine.graph.out_degree(t_id) == 0
        ]
        for end_id in end_tasks:
            D_end = self.tasks[end_id].normal_time
            prob += (
                T_finish - T[end_id] + y[end_id] >= D_end,
                f"Terminal_{end_id}"
            )

        # Target Deadline Constraint: T_finish <= target_deadline
        deadline_constraint = (T_finish <= target_deadline)
        prob += (deadline_constraint, "Target_Deadline_Constraint")

        # Solve with default solver (silently)
        solver = pulp.PULP_CBC_CMD(msg=False)
        status_code = prob.solve(solver)
        status = pulp.LpStatus[status_code]

        if status != "Optimal":
            return {
                "status": status,
                "message": f"Solver terminated with status: {status}",
                "target_deadline": target_deadline
            }

        # Extract Optimal Results
        crash_schedule = {}
        effective_durations = {}
        task_start_times = {}
        total_crashing_cost = round(pulp.value(prob.objective), 2)

        for t_id, task in self.tasks.items():
            crashed_amount = round(y[t_id].varValue, 4)
            crash_schedule[t_id] = {
                "normal_time": task.normal_time,
                "crash_time": task.crash_time,
                "crashed_amount": crashed_amount,
                "effective_duration": round(task.normal_time - crashed_amount, 4),
                "cost_slope": task.crash_cost_per_unit,
                "task_crash_cost": round(crashed_amount * task.crash_cost_per_unit, 2)
            }
            effective_durations[t_id] = task.normal_time - crashed_amount
            task_start_times[t_id] = round(T[t_id].varValue, 4)

        # Recompute CPM on crashed durations to evaluate new critical path & floats
        crashed_cpm = self.cpm_engine.compute_schedule(custom_durations=effective_durations)

        # Extract Shadow Price (Dual Value) of the deadline constraint
        shadow_price = 0.0
        if "Target_Deadline_Constraint" in prob.constraints:
            dual_val = prob.constraints["Target_Deadline_Constraint"].pi
            if dual_val is not None:
                shadow_price = round(abs(dual_val), 2)

        # Calculate Indirect and Total Costs
        indirect_cost_rate = self.schedule.indirect_cost_per_unit_time
        actual_duration = round(pulp.value(T_finish), 4)
        total_indirect_cost = round(actual_duration * indirect_cost_rate, 2)
        combined_cost = round(total_crashing_cost + total_indirect_cost, 2)

        return {
            "status": "Optimal",
            "target_deadline": target_deadline,
            "achieved_duration": actual_duration,
            "normal_duration": self.normal_project_duration,
            "min_possible_duration": self.min_project_duration,
            "total_time_saved": round(self.normal_project_duration - actual_duration, 4),
            "total_crashing_cost": total_crashing_cost,
            "indirect_cost_rate": indirect_cost_rate,
            "total_indirect_cost": total_indirect_cost,
            "combined_cost": combined_cost,
            "deadline_shadow_price": shadow_price,
            "crash_schedule": crash_schedule,
            "task_start_times": task_start_times,
            "critical_tasks": crashed_cpm["critical_tasks"],
            "critical_paths": crashed_cpm["critical_paths"],
            "schedule_details": crashed_cpm
        }