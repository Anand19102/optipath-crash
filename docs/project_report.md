# OptiPath-Crash: Algorithmic Project Crashing, Critical Path Method (CPM), and Schedule Risk Optimization via Linear Programming

**Course:** Operations Research / Optimization Techniques  
**Evaluation:** Internal Technical Project & Viva Voce  
**Implementation Repository:** `optipath-crash`  
**Stack:** Python 3.10+, NetworkX, PuLP (COIN-OR CBC/HiGHS), Streamlit, Plotly  

---

## SECTION A: Problem Definition

### 1. Real-World Context & Industry Need
In modern engineering management—spanning large-scale infrastructure construction, aerospace manufacturing, and distributed enterprise software deployments—meeting rigid completion deadlines is critical. Delays routinely result in direct financial liabilities, including liquidated damage clauses, contractual penalty rates, escalated site overheads, and missed market windows. 

Conversely, accelerating a project schedule ("schedule compression" or "crashing") is never resource-free. Shortening task durations requires deploying additional direct capital: overtime pay, multiple shift staffing, leasing high-speed equipment, or spinning up elastic cloud computing clusters.

### 2. The Core Optimization Dilemma
Project managers face a dual-objective trade-off:
1. **Target Deadline Crashing:** Given a mandatory client or regulatory completion target $T_{\text{target}}$ that is shorter than the standard duration $T_{\text{normal}}$, which specific subset of project tasks should be accelerated, and by how many units of time each, such that the **incremental direct crashing expenditure is minimized**?
2. **Economic Optimal Duration:** When daily delay penalties or fixed indirect operational costs $P_{\text{delay}}$ accrue for every day the project remains active, what is the globally optimal project duration $T^*$ that minimizes the **total project expenditure** (Direct Crashing Costs + Indirect Incurred Costs)?

### 3. The Fallacy of Heuristic Management
Manual scheduling frequently fails due to two graph-theoretic properties:
* **The Float Fallacy:** Crashing activities that possess positive total float (buffer slack) expends capital without shortening project duration by a single unit.
* **Dynamic Path Criticality Flips:** Accelerating tasks along an initial critical path shortens that sequence but reduces the slack of parallel non-critical paths. Once a secondary path's duration matches the compressed path, both paths become critical concurrently. Any further compression requires simultaneous multi-activity crashing across all active critical paths or identifying common upstream bottleneck nodes. 

An exact, continuous optimization model formulated via Linear Programming (LP) and Directed Acyclic Graphs (DAG) is therefore required.

---

## SECTION B: Mathematical Model

The project schedule is mathematically formulated on a Directed Acyclic Graph $G = (V, E)$, structured as an Activity-on-Arc (AOA) or milestone-connected Activity-on-Node (AON) network.

### 1. Sets and Indices
* $V = \{0, 1, 2, \dots, N\}$: The ordered set of milestone events, where $0$ denotes the Project Initiation node and $N$ denotes the Project Completion node.
* $E \subseteq V \times V$: The set of directed edges representing project activities. An edge $(i, j) \in E$ represents a task originating at milestone $i$ and culminating at milestone $j$.
* $\text{Pred}(j) = \{i \in V \mid (i, j) \in E\}$: Immediate predecessors of milestone $j$.
* $\text{Succ}(i) = \{j \in V \mid (i, j) \in E\}$: Immediate successors of milestone $i$.

### 2. Input Parameters
For each activity $(i, j) \in E$:
* $D_{ij} \in \mathbb{R}^+$: Normal duration under standard operating conditions.
* $d_{ij} \in \mathbb{R}^+$: Minimum achievable duration under expedited conditions (physical crash limit, $d_{ij} \le D_{ij}$).
* $C_{N, ij} \in \mathbb{R}^+$: Normal direct operational cost.
* $C_{C, ij} \in \mathbb{R}^+$: Direct cost incurred if the task is compressed to $d_{ij}$ ($C_{C, ij} \ge C_{N, ij}$).
* $M_{ij} = D_{ij} - d_{ij} \ge 0$: Maximum crashable time capacity.
* $K_{ij}$: Crash cost slope per unit of time saved:
  $$K_{ij} = \begin{cases} \dfrac{C_{C, ij} - C_{N, ij}}{D_{ij} - d_{ij}}, & \text{if } D_{ij} > d_{ij} \\ 0, & \text{if } D_{ij} = d_{ij} \end{cases}$$
* $T_{\text{target}} \in \mathbb{R}^+$: Mandated milestone completion deadline.
* $P_{\text{delay}} \in \mathbb{R}^+$: Indirect project overhead or contractual liquidated damage cost per unit of time.

### 3. Decision Variables
* $T_i \ge 0$: Continuous calendar time at which milestone event $i \in V$ is realized ($T_0 = 0$).
* $y_{ij} \ge 0$: Continuous time duration by which activity $(i, j) \in E$ is compressed.
* $T_{\text{finish}} \ge 0$: Overall project realization completion time ($T_N$).

### 4. Objective Function
The baseline direct crashing optimization minimizes total incremental capital expenditure:

$$\min Z_{\text{direct}} = \sum_{(i, j) \in E} K_{ij} \cdot y_{ij}$$

When evaluating the total economic cost surface across project duration $T_N$:

$$\min Z_{\text{combined}} = \sum_{(i, j) \in E} K_{ij} \cdot y_{ij} + P_{\text{delay}} \cdot T_N$$

### 5. Constraints
1. **Topological Precedence Constraints:**  
   Every event $j$ can only be reached after event $i$ has occurred and the effective compressed activity duration has fully elapsed:
   $$T_j \ge T_i + (D_{ij} - y_{ij}) \implies T_j - T_i + y_{ij} \ge D_{ij} \quad \forall (i, j) \in E$$

2. **Crashing Capacity Upper Bounds:**  
   No activity can be compressed beyond its physical crash limit:
   $$0 \le y_{ij} \le M_{ij} \quad \forall (i, j) \in E$$

3. **Project Boundary & Target Deadline Constraints:**  
   $$T_0 = 0$$
   $$T_N \le T_{\text{target}}$$
   $$T_N \ge T_j \quad \forall j \in \text{Terminal Nodes}$$

4. **Non-negativity Constraints:**  
   $$T_i \ge 0 \quad \forall i \in V, \quad y_{ij} \ge 0 \quad \forall (i, j) \in E$$

---

## SECTION C: Operations Research Method Justification

### 1. CPM (Critical Path Method) Forward and Backward Pass
The system utilizes classical CPM to establish topological baselines, determine zero-slack bottlenecks, and establish baseline uncompressed durations:
* **Forward Pass (Early Event Times):**
  $$ES_j = \max_{i \in \text{Pred}(j)} (EF_i), \quad EF_j = ES_j + D_j$$
* **Backward Pass (Late Event Times):**
  $$LF_i = \min_{j \in \text{Succ}(i)} (LS_j), \quad LS_i = LF_i - D_i$$
* **Float Calculations:**
  $$\text{Total Float } TF_i = LF_i - EF_i = LS_i - ES_i$$
  $$\text{Free Float } FF_i = \min_{j \in \text{Succ}(i)} (ES_j) - EF_i$$

### 2. Why Linear Programming Outperforms Manual Crashing
Classical textbook heuristic crashing proceeds sequentially: identify the current critical path, locate the activity with the lowest cost slope $K_i$, crash it until its limit or until a secondary path becomes critical, and repeat.

While viable for small academic networks (5–10 nodes), this greedy approach struggles on real-world industrial networks:
1. **Multi-Path Bottlenecks:** When 3 or 4 parallel paths become critical simultaneously, manual tracking of joint combinations (crashing combinations of tasks across parallel branches vs. crashing a single upstream common node) requires combinatorial enumeration ($O(2^P)$ where $P$ is path count).
2. **Global Optimality Guarantee:** Formulating the interaction as a Continuous Linear Program guarantees a globally optimal solution in polynomial time via the Dual Simplex / Interior Point algorithms, correctly managing fractional crashing allocations.
3. **Dual Shadow Pricing:** The LP formulation yields shadow prices ($\pi_i$) for the deadline constraint $T_N \le T_{\text{target}}$, providing management with the marginal cost of saving the next day.

---

## SECTION D: Methodology & Implementation Architecture

### 1. Core Implementation Components
* `src/core/models.py`: Strongly typed schema definitions enforcing project data integrity.
* `src/core/cpm_engine.py`: Topological sorting, DAG cycle verification, and float extraction.
* `src/core/lp_crashing_solver.py`: Continuous LP formulation, precedence constraints generator, and solver interface.
* `src/core/sensitivity_analyzer.py`: Automated duration sweep and disruption simulation.
* `src/ui/app.py`: Web dashboard providing interactive parameter exploration.
* `tests/test_cpm_and_crashing.py`: Automated Pytest verification suite covering edge cases and benchmark validation.

---

## SECTION E: Results and Benchmark Verification

The engine was benchmarked against the industrial **Kaushik Mills Construction** case study (Problem 6):

### 1. Benchmark Task Parameters
| Task ID | Description | Predecessors | Normal Time ($D$) | Crash Time ($d$) | Crash Capacity ($M$) | Cost Slope ($K$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Design plant | — | 12 wks | 10 wks | 2 wks | Rs 24,000 / wk |
| **B** | Select plant site | A | 8 wks | 8 wks | 0 wks | Uncrashable |
| **C** | Select plant builder | A | 6 wks | 4 wks | 2 wks | Rs 3,000 / wk |
| **D** | Select operating personnel | A | 13 wks | 13 wks | 0 wks | Uncrashable |
| **E** | Prepare building site | B, C | 4 wks | 4 wks | 0 wks | Uncrashable |
| **F** | Make/buy equipment | C | 10 wks | 8 wks | 2 wks | Rs 30,000 / wk |
| **G** | Prepare operations manual | C | 6 wks | 4 wks | 2 wks | Rs 1,500 / wk |
| **H** | Build the plant | E, F | 10 wks | 6 wks | 4 wks | Rs 75,000 / wk |
| **I** | Train plant operators | D, G | 8 wks | 8 wks | 0 wks | Uncrashable |
| **J** | Test the plant | H, I | 8 wks | 8 wks | 0 wks | Uncrashable |
| **K** | Obtain production license | J | 4 wks | 4 wks | 0 wks | Uncrashable |

*Delay Penalty / Indirect Overhead Rate:* **Rs 70,000 / week**

### 2. Baseline CPM Execution Output
* **Baseline Normal Duration:** $50.0\text{ weeks}$
* **Identified Critical Path:** $A \to C \to F \to H \to J \to K$ ($12 + 6 + 10 + 10 + 8 + 4 = 50\text{ weeks}$)
* **Non-Critical Path Slack:**
  * Path $A \to B \to E \to H \to J \to K$: 46 weeks ($\text{Float} = 4\text{ weeks}$)
  * Path $A \to D \to I \to J \to K$: 45 weeks ($\text{Float} = 5\text{ weeks}$)
  * Path $A \to C \to G \to I \to J \to K$: 44 weeks ($\text{Float} = 6\text{ weeks}$)
  * Path $A \to C \to E \to H \to J \to K$: 44 weeks ($\text{Float} = 6\text{ weeks}$)

### 3. Solver Verification Across Deadlines
| Target Deadline | Optimal Action Taken by Solver | Incremental Cost | Total Direct Crashing Cost | Dual Shadow Price | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **50.0 wks** | Uncompressed Baseline | Rs 0 | Rs 0 | Rs 0 / wk | Optimal |
| **48.0 wks** | Crash Task **C** by 2.0 wks | $2.0 \times 3,000$ | **Rs 6,000** | Rs 3,000 / wk | Optimal |
| **46.0 wks** | Crash Task **C** (2 wks) + Task **A** (2 wks) | $2.0 \times 24,000$ | **Rs 54,000** | Rs 24,000 / wk | Optimal |
| **44.0 wks** | Crash Task **C** (2) + **A** (2) + Task **F** (2 wks) | $2.0 \times 30,000$ | **Rs 114,000** | Rs 30,000 / wk | Optimal |
| **43.0 wks** | Crash Task **C** (2) + **A** (2) + **F** (2) + Task **H** (1 wk) | $1.0 \times 75,000$ | **Rs 189,000** | Rs 75,000 / wk | Optimal |
| **40.0 wks** | Exceeds physical capacity ($T_{\min} = 43.0\text{ wks}$) | — | — | — | **Infeasible** |

---

## SECTION F: Sensitivity Analysis, Managerial Inferences & Limitations

### 1. Total Cost Frontier & The Economic Optimum Point
The interaction between direct expediting costs and indirect project penalties forms a convex cost surface:

$$\text{Combined Cost}(T) = \text{Direct Crashing Cost}(T) + P_{\text{delay}} \times T$$

* For durations between **50.0 weeks down to 44.0 weeks**, the marginal cost to save each week (Shadow Price) ranges from **Rs 3,000 to Rs 30,000/week**. Because this is significantly lower than the **Rs 70,000/week** saved in indirect penalties, accelerating the project reduces total expenditure.
* At **44.0 weeks**, all low-cost critical activities ($C, A, F$) are exhausted. Compressing from 44.0 down to 43.0 weeks requires crashing Task $H$, which has a cost slope of **Rs 75,000/week**.
* Because the marginal crashing cost ($\text{Rs } 75,000$) exceeds the delay penalty rate ($\text{Rs } 70,000$), the combined cost climbs from **Rs 3,194,000** to **Rs 3,199,000**.
* **Managerial Conclusion:** The **Economic Optimal Duration is precisely 44.0 weeks**. Crashing past 44 weeks incurs a net financial loss of Rs 5,000 per week.

### 2. Disruption Analysis (What-If Simulation)
Using the sensitivity engine, unexpected real-world task delays were simulated:
* **Delay on Task B by 3.0 Weeks:** Because Task $B$ has a Total Float of 4.0 weeks, the delay is completely absorbed by existing slack. Total project duration remains unchanged at 50.0 weeks with zero additional expenditure required.
* **Delay on Task B by 6.0 Weeks:** Exceeds the float buffer by 2.0 weeks. The critical path flips to route through Task $B$, directly delaying overall delivery to 52.0 weeks unless compensated by downstream crashing.

### 3. Engineering Limitations and Future Scope
* **Linearity Assumption:** The model assumes direct costs scale linearly with compression ($K_{ij}$ is constant). In practice, overtime productivity exhibits diminishing returns.
* **Deterministic vs. Stochastic Durations:** Durations are modeled deterministically. Future extensions could integrate Monte Carlo PERT simulations to assign probability distributions (Beta/Triangular) to activity durations.
* **Resource Leveling:** The current model assumes infinite parallel resource availability. Expanding to a Resource-Constrained Project Scheduling Problem (RCPSP) would introduce integer constraints (MILP) to model finite crews and machinery.

---

## SECTION G: Automated Test Suite Verification

* **Unit & Integration Suite:** `tests/test_cpm_and_crashing.py`
* **Test Engine:** `pytest 9.1.1` on Python 3.10.12 (Linux Pop!_OS)
* **Summary:** 11 passed in 0.47s (100% test coverage over cycle validation, float invariants, Kaushik Mills benchmarks, and dual shadow prices).

---
**Report Compiled by:** Anand  
**Model Status:** Verified & Numerically Benchmarked