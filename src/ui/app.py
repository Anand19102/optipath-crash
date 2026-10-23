import os
import sys

# Guarantee repository root is bound to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from src.core.models import ProjectSchedule
from src.core.cpm_engine import CPMEngine
from src.core.lp_crashing_solver import LPCrashingSolver
from src.core.sensitivity_analyzer import SensitivityAnalyzer

# --- Application Configuration ---
st.set_page_config(
    page_title="OptiPath-Crash | Schedule Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Executive Enterprise Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #090d16;
        --card-bg: #0f172a;
        --border-color: #1e293b;
        --accent-blue: #38bdf8;
        --accent-red: #ef4444;
        --accent-green: #10b981;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: var(--bg-main);
    }

    /* Executive Hero Header */
    .dashboard-header {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }

    .project-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #ffffff;
        margin: 0 0 6px 0;
    }

    .project-subtitle {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0 0 16px 0;
        line-height: 1.5;
    }

    .meta-tag-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }

    .meta-tag {
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 500;
        background: rgba(15, 23, 42, 0.8);
        color: #cbd5e1;
        border: 1px solid #334155;
    }

    .meta-tag.accent {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.35);
    }

    .meta-tag.alert {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.35);
    }

    /* High-Precision Metric Cards */
    div[data-testid="metric-container"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    div[data-testid="metric-container"]:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }

    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
        color: #f8fafc !important;
    }

    /* Status and Alert Containers */
    .status-banner {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .status-banner.warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
    }

    .status-banner.success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
    }

    .status-banner.danger {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #f87171;
    }

    /* Polished Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 18px;
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-color: #334155 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_benchmark(filepath: str) -> dict:
    full_path = os.path.join(PROJECT_ROOT, filepath)
    with open(full_path, "r") as f:
        return json.load(f)


# ==============================================================================
# SIDEBAR CONTROL DECK
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 18px 0;">
        <span style="font-size: 0.72rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.1em; text-transform: uppercase;">Operations Research</span>
        <h3 style="margin: 2px 0 0 0; font-size: 1.35rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">OptiPath-Crash</h3>
    </div>
    """, unsafe_allow_html=True)

    benchmark_selection = st.selectbox(
        "Case Study Benchmark",
        [
            "Kaushik Mills Construction (Industrial Baseline)",
            "Enterprise Cloud Cutover (Tech Stack)",
            "Custom Upload (.JSON)"
        ]
    )

    payload = None
    if "Kaushik Mills" in benchmark_selection:
        payload = load_benchmark("data/kaushik_mills.json")
    elif "Cloud Cutover" in benchmark_selection:
        payload = load_benchmark("data/cloud_migration.json")
    else:
        uploaded = st.file_uploader("Upload Project JSON Schema", type=["json"])
        if uploaded is not None:
            payload = json.load(uploaded)

    if not payload:
        st.info("Awaiting valid project schedule definition.")
        st.stop()

    schedule = ProjectSchedule(**payload)
    time_unit = schedule.time_unit
    solver = LPCrashingSolver(schedule)

    t_norm = solver.normal_project_duration
    t_min = solver.min_project_duration

    st.markdown("---")
    st.markdown("<span style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Optimization Constraints</span>", unsafe_allow_html=True)

    target_deadline = st.slider(
        f"Target Deadline ({time_unit})",
        min_value=float(t_min),
        max_value=float(t_norm),
        value=float(t_norm),
        step=0.5,
        help="Contractual target completion time to compress toward."
    )

    delay_rate = st.number_input(
        f"Delay Penalty / Indirect Rate (Rs/{time_unit})",
        min_value=0.0,
        value=float(schedule.indirect_cost_per_unit_time),
        step=5000.0,
        help="Contractual liquidated damages or ongoing indirect operational overhead accrued per unit of time."
    )
    schedule.indirect_cost_per_unit_time = delay_rate

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748b; line-height: 1.6;">
        <b>Engine:</b> Topological CPM + Continuous LP<br>
        <b>Solver:</b> COIN-OR CBC (Simplex Exact)<br>
        <b>Formulation:</b> Continuous Precedence Dual LP
    </div>
    """, unsafe_allow_html=True)

# Execute LP Optimization
solve_res = solver.solve_crashing(target_deadline=target_deadline)
cpm_data = solve_res["schedule_details"]

# ==============================================================================
# EXECUTIVE HEADER & STATS
# ==============================================================================
st.markdown(f"""
<div class="dashboard-header">
    <div class="project-title">{schedule.project_name}</div>
    <div class="project-subtitle">Algorithmic Critical Path Method (CPM) & Continuous Linear Programming Schedule Risk Optimization</div>
    <div class="meta-tag-container">
        <span class="meta-tag accent">Normal Duration: {t_norm:.1f} {time_unit}</span>
        <span class="meta-tag alert">Physical Crash Bound: {t_min:.1f} {time_unit}</span>
        <span class="meta-tag">Active Schedule: {solve_res['achieved_duration']:.1f} {time_unit}</span>
        <span class="meta-tag">Penalty Rate: Rs {delay_rate:,.0f}/{time_unit}</span>
        <span class="meta-tag">Activities: {len(schedule.tasks)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Executive Primary KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    time_saved = solve_res["total_time_saved"]
    st.metric(
        "Project Schedule",
        f"{solve_res['achieved_duration']:.1f} {time_unit}",
        f"-{time_saved:.1f} {time_unit} Compressed" if time_saved > 0 else "Baseline Schedule"
    )
with c2:
    st.metric(
        "Direct Crashing Outlay",
        f"Rs {solve_res['total_crashing_cost']:,.0f}",
        f"+Rs {solve_res['total_crashing_cost']:,.0f}" if solve_res['total_crashing_cost'] > 0 else "Rs 0"
    )
with c3:
    st.metric(
        "Aggregate Project Cost",
        f"Rs {solve_res['combined_cost']:,.0f}",
        f"Rs {solve_res['total_indirect_cost']:,.0f} Indirect"
    )
with c4:
    st.metric(
        f"Marginal Crash Cost (π)",
        f"Rs {solve_res['deadline_shadow_price']:,.0f}/{time_unit}",
        "Dual Variable Value" if solve_res['deadline_shadow_price'] > 0 else "Slack Free"
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Dynamic Critical Path Multiplicity Alert
active_paths = solve_res["critical_paths"]
if len(active_paths) > 1:
    st.markdown(f"""
    <div class="status-banner warning">
        <b>Structural Bottleneck Warning:</b> Exactly <b>{len(active_paths)} parallel paths</b> are concurrently critical. 
        Any subsequent schedule compression requires multi-activity co-crashing across all active branches.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN WORKSPACE TABS
# ==============================================================================
tabs = st.tabs([
    "Activity Compression & Schedule",
    "Topological DAG Network",
    "Cost Frontier & Pareto Analysis",
    "What-If Disruption Sandbox"
])

# ------------------------------------------------------------------------------
# TAB 1: SCHEDULE & GANTT
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Activity Timeline & Schedule Compression Breakdown")
    st.caption("Inspect early/late event times, task float buffers, and exact LP-allocated compression amounts.")

    table_data = []
    for t in schedule.tasks:
        c_info = solve_res["crash_schedule"][t.id]
        is_crit = t.id in solve_res["critical_tasks"]
        table_data.append({
            "Task ID": t.id,
            "Description": t.description,
            "Predecessors": ", ".join(t.predecessors) if t.predecessors else "—",
            "Normal (D)": f"{t.normal_time:.1f}",
            "Crash Limit (d)": f"{t.crash_time:.1f}",
            "Compressed (y)": f"{c_info['crashed_amount']:.1f}",
            "Effective (D')": f"{c_info['effective_duration']:.1f}",
            "Cost Slope (K)": f"Rs {t.crash_cost_per_unit:,.0f}",
            "Crash Outlay": f"Rs {c_info['task_crash_cost']:,.0f}",
            "Early Start (ES)": f"{cpm_data['es'][t.id]:.1f}",
            "Early Finish (EF)": f"{cpm_data['ef'][t.id]:.1f}",
            "Total Float": f"{cpm_data['total_float'][t.id]:.1f}",
            "Status": "CRITICAL" if is_crit else "Positive Slack"
        })

    df_display = pd.DataFrame(table_data)

    def highlight_critical(row):
        return ['background-color: rgba(239, 68, 68, 0.12); color: #fca5a5' if row["Status"] == "CRITICAL" else '' for _ in row]

    st.dataframe(
        df_display.style.apply(highlight_critical, axis=1),
        use_container_width=True,
        hide_index=True
    )

    csv_schedule = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Current Schedule Plan (CSV)",
        data=csv_schedule,
        file_name=f"{schedule.project_name.lower().replace(' ', '_')}_schedule.csv",
        mime="text/csv"
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Coordinated Four-Segment Gantt Schedule")

    gantt_rows = []
    for t in schedule.tasks:
        es = cpm_data["es"][t.id]
        dur = solve_res["crash_schedule"][t.id]["effective_duration"]
        crashed = solve_res["crash_schedule"][t.id]["crashed_amount"]
        tf = cpm_data["total_float"][t.id]
        is_crit = t.id in solve_res["critical_tasks"]

        # 1. Main active duration bar
        gantt_rows.append({
            "Task": f"<b>{t.id}</b>: {t.description}",
            "Start": es,
            "Duration": dur,
            "Segment": "Critical Execution" if is_crit else "Standard Execution"
        })
        # 2. Crashed portion (time compressed)
        if crashed > 0:
            gantt_rows.append({
                "Task": f"<b>{t.id}</b>: {t.description}",
                "Start": es + dur,
                "Duration": crashed,
                "Segment": "Expedited Compression"
            })
        # 3. Available Total Float buffer (slack)
        if tf > 0:
            gantt_rows.append({
                "Task": f"<b>{t.id}</b>: {t.description}",
                "Start": es + dur + crashed,
                "Duration": tf,
                "Segment": "Available Float Buffer"
            })

    df_gantt = pd.DataFrame(gantt_rows)
    fig_gantt = px.bar(
        df_gantt,
        base="Start",
        x="Duration",
        y="Task",
        color="Segment",
        orientation="h",
        color_discrete_map={
            "Critical Execution": "#ef4444",
            "Standard Execution": "#0284c7",
            "Expedited Compression": "#f59e0b",
            "Available Float Buffer": "#334155"
        },
        labels={"x": f"Project Timeline ({time_unit})", "Task": ""},
    )

    fig_gantt.update_layout(
        template="plotly_dark",
        plot_bgcolor="#090d16",
        paper_bgcolor="#090d16",
        height=460,
        margin=dict(l=10, r=20, t=10, b=30),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, family="Inter")),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(size=11, family="JetBrains Mono")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TOPOLOGICAL NETWORK DAG
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Topological Activity-on-Node (AON) Network")
    st.caption("Stage-ordered milestone layout derived directly from topological dependency passes. Flow moves strictly left to right.")

    G = solver.cpm_engine.graph
    topo_order = list(nx.topological_sort(G))

    # Stage level assignment for clean left-to-right layering
    stages = {}
    for node in topo_order:
        preds = list(G.predecessors(node))
        stages[node] = 0 if not preds else max(stages[p] for p in preds) + 1

    stage_groups = {}
    for node, stg in stages.items():
        stage_groups.setdefault(stg, []).append(node)

    node_pos = {}
    for stg, nodes in stage_groups.items():
        count = len(nodes)
        for idx, node in enumerate(nodes):
            y_coord = (idx - (count - 1) / 2.0) * 1.8
            node_pos[node] = (stg * 3.0, y_coord)

    crit_set = set(solve_res["critical_tasks"])
    fig_dag = go.Figure()

    # Draw edge lines and directional vector arrowheads
    for u, v in G.edges():
        x0, y0 = node_pos[u]
        x1, y1 = node_pos[v]
        is_crit_edge = (u in crit_set and v in crit_set)
        color = "#ef4444" if is_crit_edge else "#334155"
        width = 2.8 if is_crit_edge else 1.2

        fig_dag.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="none",
            showlegend=False
        ))

        # Vector Arrowheads
        vx = x1 - x0
        vy = y1 - y0
        v_len = np.hypot(vx, vy)
        if v_len > 0:
            ux, uy = vx / v_len, vy / v_len
            ax = x1 - ux * 0.45
            ay = y1 - uy * 0.45
            fig_dag.add_annotation(
                x=ax, y=ay, ax=ax - ux * 0.2, ay=ay - uy * 0.2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=width,
                arrowcolor=color
            )

    # Render Node Traces
    node_x, node_y, node_color, node_text, node_hover = [], [], [], [], []
    for node in G.nodes():
        x, y = node_pos[node]
        node_x.append(x)
        node_y.append(y)
        is_crit = node in crit_set
        node_color.append("#ef4444" if is_crit else "#0284c7")
        node_text.append(f"<b>{node}</b>")

        c_amt = solve_res["crash_schedule"][node]["crashed_amount"]
        eff_dur = solve_res["crash_schedule"][node]["effective_duration"]
        tf = cpm_data["total_float"][node]
        desc = schedule.tasks[[t.id for t in schedule.tasks].index(node)].description

        node_hover.append(
            f"<b>Activity {node}</b>: {desc}<br>"
            f"Effective Duration: {eff_dur:.1f} {time_unit}<br>"
            f"Compressed: {c_amt:.1f} {time_unit}<br>"
            f"Total Float: {tf:.1f} {time_unit}<br>"
            f"Status: {'CRITICAL (Zero Slack)' if is_crit else 'Non-Critical'}"
        )

    fig_dag.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(family="JetBrains Mono", size=12, color="#ffffff"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=36,
            color=node_color,
            line=dict(width=2, color="#ffffff")
        ),
        showlegend=False
    ))

    fig_dag.update_layout(
        template="plotly_dark",
        plot_bgcolor="#090d16",
        paper_bgcolor="#090d16",
        height=480,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig_dag, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: COST FRONTIER & PARETO
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Time-Cost Pareto Frontier & Dual Optimization Surface")
    st.caption("Continuous parametric sweep of project durations from normal baseline down to physical crash limits.")

    analyzer = SensitivityAnalyzer(schedule)
    sweep_data = analyzer.run_deadline_sweep(step_size=0.5)
    df_sweep = sweep_data["sweep_table"]
    opt = sweep_data["optimal_duration_point"]

    fig_frontier = go.Figure()

    # Direct Outlay Curve
    fig_frontier.add_trace(go.Scatter(
        x=df_sweep["target_deadline"],
        y=df_sweep["direct_crashing_cost"],
        mode="lines+markers",
        name="Direct Expediting Cost",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=5)
    ))

    # Indirect Delay Penalty Curve
    fig_frontier.add_trace(go.Scatter(
        x=df_sweep["target_deadline"],
        y=df_sweep["indirect_cost"],
        mode="lines+markers",
        name="Indirect / Delay Liability",
        line=dict(color="#06b6d4", width=2, dash="dash"),
        marker=dict(size=5)
    ))

    # Aggregate Cost Curve
    fig_frontier.add_trace(go.Scatter(
        x=df_sweep["target_deadline"],
        y=df_sweep["combined_cost"],
        mode="lines+markers",
        name="Total Combined Project Cost",
        line=dict(color="#10b981", width=3.5),
        marker=dict(size=8, symbol="diamond")
    ))

    # Optimal Inflection Marker
    if opt:
        fig_frontier.add_vline(
            x=opt["target_deadline"],
            line_width=1.5,
            line_dash="dot",
            line_color="#ef4444",
            annotation_text=f"Global Optimum: {opt['target_deadline']} {time_unit}",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#ef4444", family="JetBrains Mono")
        )

    fig_frontier.update_layout(
        template="plotly_dark",
        plot_bgcolor="#090d16",
        paper_bgcolor="#090d16",
        height=480,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis=dict(
            title=f"Overall Project Duration ({time_unit})",
            gridcolor="#1e293b",
            tickfont=dict(family="JetBrains Mono", size=11)
        ),
        yaxis=dict(
            title="Total Expenditure (Rs)",
            gridcolor="#1e293b",
            tickfont=dict(family="JetBrains Mono", size=11)
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    col_inf1, col_inf2 = st.columns([2.5, 1])
    with col_inf1:
        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-left: 3px solid #10b981; padding: 16px 20px; border-radius: 8px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Analytic Minimum Assessment</div>
            <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">
                The unconstrained economic optimum rests precisely at <b>{opt['target_deadline']} {time_unit}</b> with a minimized combined expenditure of <b>Rs {opt['combined_cost']:,.0f}</b>. 
                Accelerating project execution beyond this threshold demands activities whose marginal cost slopes exceed the statutory penalty threshold of <b>Rs {delay_rate:,.0f}/{time_unit}</b>, resulting in diminishing returns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_inf2:
        csv_frontier = df_sweep.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Frontier Data (CSV)",
            data=csv_frontier,
            file_name=f"{schedule.project_name.lower().replace(' ', '_')}_pareto_frontier.csv",
            mime="text/csv",
            use_container_width=True
        )

# ------------------------------------------------------------------------------
# TAB 4: WHAT-IF DISRUPTION SANDBOX
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Schedule Disruption & Buffer Sensitivity Simulator")
    st.caption("Simulate unexpected task slippage to determine whether float buffers absorb the disruption or trigger delivery delays.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        target_task = st.selectbox("Select Target Activity", [t.id for t in schedule.tasks])
    with col_sim2:
        injected_slip = st.slider(
            f"Simulated Delay Amount ({time_unit})",
            min_value=0.0,
            max_value=12.0,
            value=2.0,
            step=0.5
        )

    disruption_res = analyzer.evaluate_task_delay_impact(task_id=target_task, delay_amount=injected_slip)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Task Total Float (Slack Buffer)", f"{disruption_res['original_total_float']:.1f} {time_unit}")
    with m2:
        st.metric("Disruption Absorbed by Slack", "TRUE (Protected)" if disruption_res["absorbed_by_slack"] else "FALSE (Critical Flip)")
    with m3:
        st.metric("Net Milestone Slippage", f"{disruption_res['project_delay_incurred']:.1f} {time_unit}")

    if disruption_res["absorbed_by_slack"]:
        st.markdown(f"""
        <div class="status-banner success">
            <b>Buffer Protected:</b> The injected slippage of {injected_slip:.1f} {time_unit} on Task {target_task} 
            is fully buffered by the activity's total float. Final delivery remains on schedule at {disruption_res['new_project_duration']:.1f} {time_unit}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner danger">
            <b>Critical Slippage Incurred:</b> The injected slippage exceeds available float buffers by {disruption_res['project_delay_incurred']:.1f} {time_unit}. 
            The completion milestone slips to {disruption_res['new_project_duration']:.1f} {time_unit}.
        </div>
        """, unsafe_allow_html=True)