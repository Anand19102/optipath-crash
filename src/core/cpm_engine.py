from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from src.core.models import ProjectSchedule, Task

class CPMEngine:
    def __init__(self, schedule: ProjectSchedule):
        self.schedule = schedule
        self.tasks: Dict[str, Task] = {t.id: t for t in schedule.tasks}
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        for task_id, task in self.tasks.items():
            self.graph.add_node(task_id, duration=task.normal_time, task=task)

        for task_id, task in self.tasks.items():
            for pred in task.predecessors:
                if pred not in self.tasks:
                    raise ValueError(f"Predecessor '{pred}' for task '{task_id}' does not exist.")
                self.graph.add_edge(pred, task_id)

        # Validate DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Cyclic dependency detected in project schedule: {cycles}")

    def compute_schedule(self, custom_durations: Optional[Dict[str, float]] = None) -> Dict:
        """
        Executes Forward & Backward pass to compute ES, EF, LS, LF, Float, and Critical Path.
        Allows custom durations (e.g., crashed durations).
        """
        durations = custom_durations or {t_id: t.normal_time for t_id, t in self.tasks.items()}
        topo_order = list(nx.topological_sort(self.graph))

        # Forward Pass: ES, EF
        es: Dict[str, float] = {}
        ef: Dict[str, float] = {}

        for node in topo_order:
            preds = list(self.graph.predecessors(node))
            if not preds:
                es[node] = 0.0
            else:
                es[node] = max(ef[p] for p in preds)
            ef[node] = es[node] + durations[node]

        total_duration = max(ef.values()) if ef else 0.0

        # Backward Pass: LS, LF
        ls: Dict[str, float] = {}
        lf: Dict[str, float] = {}

        for node in reversed(topo_order):
            succs = list(self.graph.successors(node))
            if not succs:
                lf[node] = total_duration
            else:
                lf[node] = min(ls[s] for s in succs)
            ls[node] = lf[node] - durations[node]

        # Float Calculations
        total_float: Dict[str, float] = {}
        free_float: Dict[str, float] = {}
        critical_path_tasks: List[str] = []

        for node in topo_order:
            tf = round(lf[node] - ef[node], 4)
            total_float[node] = tf

            succs = list(self.graph.successors(node))
            if not succs:
                ff = round(total_duration - ef[node], 4)
            else:
                ff = round(min(es[s] for s in succs) - ef[node], 4)
            free_float[node] = ff

            if abs(tf) < 1e-5:
                critical_path_tasks.append(node)

        # Extract all critical paths (complete chains)
        start_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        end_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        
        all_critical_paths = []
        for s in start_nodes:
            for e in end_nodes:
                for path in nx.all_simple_paths(self.graph, s, e):
                    if all(p in critical_path_tasks for p in path):
                        path_len = sum(durations[p] for p in path)
                        if abs(path_len - total_duration) < 1e-5:
                            all_critical_paths.append(path)

        return {
            "total_duration": total_duration,
            "es": es,
            "ef": ef,
            "ls": ls,
            "lf": lf,
            "total_float": total_float,
            "free_float": free_float,
            "critical_tasks": critical_path_tasks,
            "critical_paths": all_critical_paths
        }
