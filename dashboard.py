"""
NIC-CloudBot: Interactive Autonomous SRE & Cloud Resource Scaling Operations Center
Enterprise Viva Demonstration Edition with Visual Kubernetes Topology & 3-Act Story Mode.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import random
from datetime import datetime
import importlib
import agent_graph
importlib.reload(agent_graph)

from agent_graph import (
    build_cloudbot_graph,
    get_expanded_incident_catalog,
    generate_dynamic_random_incident
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NIC-CloudBot SRE Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark-Mode CSS with Visual Pod Badges
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .k8s-node-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        min-height: 220px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .k8s-node-box-cordon {
        background: #1c130d !important;
        border: 2px dashed #f97316 !important;
        border-radius: 10px;
        padding: 16px;
        min-height: 220px;
        margin-bottom: 12px;
    }
    .node-header {
        font-weight: 700;
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .pod-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-family: monospace;
        margin: 4px;
        font-weight: 600;
    }
    .pod-green {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    .pod-red {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        animation: pulse 1.5s infinite;
    }
    .pod-yellow {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
    }
    .pod-blue {
        background-color: #1e3a8a;
        color: #60a5fa;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    .cot-container {
        background-color: #030712;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        border-radius: 6px;
        line-height: 1.6;
        color: #e2e8f0;
    }
    .story-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "cluster_state" not in st.session_state:
    st.session_state.cluster_state = {
        "hdfs_replicas": 3,
        "web_replicas": 4,
        "cpu_util": 34.0,
        "mem_util": 45.0,
        "qps": 1200,
        "cluster_health": "HEALTHY",
        "cordoned_nodes": []
    }

if "story_phase" not in st.session_state:
    st.session_state.story_phase = "1_STEADY"  # 1_STEADY, 2_CHAOS, 3_DIAGNOSING, 4_RESOLVED

if "current_incident" not in st.session_state:
    st.session_state.current_incident = None

if "hitl_queue" not in st.session_state:
    st.session_state.hitl_queue = []

if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

@st.cache_resource
def get_graph():
    return build_cloudbot_graph()

app = get_graph()
incident_catalog = get_expanded_incident_catalog()


# ---------------------------------------------------------------------------
# Top Navigation & High-Level Telemetry
# ---------------------------------------------------------------------------
st.title("🛡️ NIC-CloudBot: Autonomous SRE & Cloud Scaling Center")
st.caption("Compound AI Architecture • 5-Agent LangGraph StateGraph • Fine-Tuned Mistral-7B SLM • Zero-GPU Fast Perception")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    h_status = st.session_state.cluster_state["cluster_health"]
    delta_color = "normal" if "HEALTHY" in h_status else "inverse"
    st.metric("Cluster Health", h_status, delta="Nominal" if "HEALTHY" in h_status else "Action Required", delta_color=delta_color)
with col2:
    st.metric("DataNode Pods", f"{st.session_state.cluster_state['hdfs_replicas']} Pods", delta=f"{st.session_state.cluster_state['hdfs_replicas']-3:+d}" if st.session_state.cluster_state['hdfs_replicas'] != 3 else "Baseline")
with col3:
    st.metric("Web Ingress Pods", f"{st.session_state.cluster_state['web_replicas']} Pods", delta=f"{st.session_state.cluster_state['web_replicas']-4:+d}" if st.session_state.cluster_state['web_replicas'] != 4 else "Baseline")
with col4:
    st.metric("Aggregate CPU", f"{st.session_state.cluster_state['cpu_util']:.1f}%")
with col5:
    st.metric("HITL Approvals", len(st.session_state.hitl_queue), delta=f"{len(st.session_state.hitl_queue)} Pending", delta_color="inverse" if len(st.session_state.hitl_queue) > 0 else "off")

st.divider()


# ---------------------------------------------------------------------------
# Sidebar: Chaos Engineering & Live Trigger Panel
# ---------------------------------------------------------------------------
st.sidebar.header("⚡ Chaos Engineering & Scenarios")
st.sidebar.caption("Simulate production cloud incidents to demonstrate the multi-agent system:")

selected_key = st.sidebar.selectbox("Select Incident Scenario (8 Available):", list(incident_catalog.keys()))
active_incident = incident_catalog[selected_key]

# Dynamic Randomizer Button
if st.sidebar.button("🎲 Generate Random Dynamic Incident", use_container_width=True):
    active_incident = generate_dynamic_random_incident()
    st.session_state.current_incident = active_incident
    st.sidebar.success(f"Generated: {active_incident['title']}")

# Interactive "What-If" Capacity Sliders
st.sidebar.divider()
st.sidebar.subheader("🎛️ Interactive 'What-If' Sliders")
st.sidebar.caption("Adjust real-time load to test the SLM's dynamic HPA scaling formula:")
custom_qps = st.sidebar.slider("Query Load (QPS):", min_value=1000, max_value=20000, value=active_incident["qps"], step=500)
custom_cpu = st.sidebar.slider("Cluster CPU Saturation (%):", min_value=30, max_value=99, value=active_incident["cpu"], step=1)

# Primary Story Trigger Buttons
st.sidebar.divider()
trigger_col1, trigger_col2 = st.sidebar.columns(2)
with trigger_col1:
    run_clicked = st.button("🚨 Inject Incident", use_container_width=True, type="primary")
with trigger_col2:
    reset_clicked = st.button("🔄 Reset Baseline", use_container_width=True)

if reset_clicked:
    st.session_state.cluster_state = {
        "hdfs_replicas": 3,
        "web_replicas": 4,
        "cpu_util": 34.0,
        "mem_util": 45.0,
        "qps": 1200,
        "cluster_health": "HEALTHY",
        "cordoned_nodes": []
    }
    st.session_state.story_phase = "1_STEADY"
    st.session_state.hitl_queue = []
    st.session_state.last_result = None
    st.session_state.current_incident = None
    st.rerun()


# ---------------------------------------------------------------------------
# Incident Processing Execution
# ---------------------------------------------------------------------------
if run_clicked:
    inc = st.session_state.current_incident if st.session_state.current_incident else active_incident
    st.session_state.current_incident = inc
    
    # Act 2: Chaos Injected
    st.session_state.story_phase = "2_CHAOS"
    st.session_state.cluster_state["cpu_util"] = custom_cpu
    st.session_state.cluster_state["qps"] = custom_qps
    st.session_state.cluster_state["cluster_health"] = "INCIDENT IN PROGRESS"

    t0 = time.time()
    state_input = {
        "raw_payload": inc["payload"],
        "log_source": inc["source"],
        "session_id": inc["session_id"],
        "event_sequence": inc["events"],
        "latency_ms": inc["latency"],
        "telemetry": {
            "cpu_percent": custom_cpu,
            "memory_percent": inc["mem"],
            "qps": custom_qps,
            "active_replicas": st.session_state.cluster_state["hdfs_replicas"] if inc["source"] == "HDFS" else st.session_state.cluster_state["web_replicas"]
        },
        "audit_trail": []
    }

    # Execute Multi-Agent Graph
    output_state = app.invoke(state_input)
    st.session_state.last_result = output_state
    st.session_state.story_phase = "4_RESOLVED"

    # Process Scaling or HITL Outcomes
    exec_status = output_state.get("execution_status")
    action = output_state.get("action_details", {})
    
    if exec_status == "AUTO_EXECUTED":
        if action.get("deployment") == "hdfs-datanode":
            st.session_state.cluster_state["hdfs_replicas"] = action.get("target_replicas", 5)
        elif action.get("deployment") == "web-frontend-service":
            st.session_state.cluster_state["web_replicas"] = action.get("target_replicas", 8)
        st.session_state.cluster_state["cluster_health"] = "HEALTHY (AUTO-SCALED)"
        st.session_state.cluster_state["cpu_util"] = 48.0  # Normalizes after scaling
        
    elif exec_status == "PENDING_HUMAN_APPROVAL":
        st.session_state.cluster_state["cluster_health"] = "DEGRADED (HITL APPROVAL REQUIRED)"
        if action.get("command") == "DRAIN_AND_MAINTAIN_NODE":
            st.session_state.cluster_state["cordoned_nodes"].append(action.get("target_node", "storage-node-04"))
        
        st.session_state.hitl_queue.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "task": action.get("command", "MANUAL_INSPECTION"),
            "target_node": action.get("target_node", "storage-node-04"),
            "reason": action.get("reason", "Critical hardware exception"),
            "state_ref": output_state
        })
    else:
        st.session_state.cluster_state["cluster_health"] = "HEALTHY"
        st.session_state.cluster_state["cpu_util"] = 34.0

    # Record to historical audit
    st.session_state.audit_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "session_id": output_state["session_id"],
        "source": output_state["log_source"],
        "status": output_state["detection_status"],
        "confidence": f"{output_state.get('anomaly_confidence', 0.0)*100:.1f}%",
        "action": action.get("action"),
        "risk_tier": output_state.get("risk_tier"),
        "decision": exec_status,
        "runtime": f"{time.time() - t0:.3f}s"
    })
    st.rerun()


# ---------------------------------------------------------------------------
# ACT 1 & ACT 2: Interactive Story Banner
# ---------------------------------------------------------------------------
current_inc = st.session_state.current_incident if st.session_state.current_incident else active_incident

st.markdown(f"""
<div class='story-card'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <span style='background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;'>STORY MODE ACTIVE</span>
            <h3 style='margin: 6px 0 2px 0;'>{current_inc['title']}</h3>
            <p style='color: #cbd5e1; margin: 0; font-size: 13.5px;'>{current_inc['story']}</p>
        </div>
        <div style='text-align: right;'>
            <span style='font-size: 12px; color: #94a3b8;'>Expected Action:</span><br>
            <b style='color: #38bdf8;'>{current_inc['expected_action']}</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VISUAL KUBERNETES CLUSTER TOPOLOGY MAP
# ---------------------------------------------------------------------------
st.subheader("🗺️ Live Kubernetes Cluster Topology & Node Architecture")

hdfs_count = st.session_state.cluster_state["hdfs_replicas"]
web_count = st.session_state.cluster_state["web_replicas"]
last_res = st.session_state.last_result
phase = st.session_state.story_phase

k_col1, k_col2, k_col3, k_col4 = st.columns(4)

# Node 1: Web Ingress Master
with k_col1:
    is_web_surge = last_res and last_res.get("log_source") == "APACHE" and "504" in last_res.get("raw_payload", "")
    st.markdown("""
    <div class='k8s-node-box'>
        <div class='node-header'>
            <span>🖥️ worker-node-01</span>
            <span style='color: #38bdf8;'>Ingress Tier</span>
        </div>
        <div style='color: #94a3b8; font-size: 11px; margin-bottom: 8px;'>Role: Nginx / Apache Reverse Proxy</div>
    """, unsafe_allow_html=True)
    
    # Render Web Pods
    for i in range(1, web_count + 1):
        if is_web_surge and i > 4:
            st.markdown(f"<span class='pod-pill pod-blue'>🔵 web-frontend-pod-0{i} (NEW)</span>", unsafe_allow_html=True)
        elif is_web_surge and i <= 2:
            st.markdown(f"<span class='pod-pill pod-red'>🔴 web-frontend-pod-0{i} (504 SATURATED)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='pod-pill pod-green'>🟢 web-frontend-pod-0{i} (RUNNING)</span>", unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# Node 2: Worker Compute Tier
with k_col2:
    is_oom = last_res and "OOM" in last_res.get("raw_payload", "")
    st.markdown("""
    <div class='k8s-node-box'>
        <div class='node-header'>
            <span>🖥️ worker-node-02</span>
            <span style='color: #a855f7;'>Compute Tier</span>
        </div>
        <div style='color: #94a3b8; font-size: 11px; margin-bottom: 8px;'>Role: Microservice JVM / App Containers</div>
    """, unsafe_allow_html=True)
    
    if is_oom:
        st.markdown("<span class='pod-pill pod-red'>🔴 java-worker-4819 (OOM KILLED)</span>", unsafe_allow_html=True)
        st.markdown("<span class='pod-pill pod-yellow'>🟡 queue-consumer-01 (GC STALL)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pod-pill pod-green'>🟢 java-worker-pod-01 (RUNNING)</span>", unsafe_allow_html=True)
        st.markdown("<span class='pod-pill pod-green'>🟢 queue-consumer-01 (RUNNING)</span>", unsafe_allow_html=True)
        st.markdown("<span class='pod-pill pod-green'>🟢 analytics-worker-01 (RUNNING)</span>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# Node 3: Storage Node-03 (Primary DataNode)
with k_col3:
    is_hdfs_socket = last_res and "E4" in last_res.get("suspicious_events", [])
    st.markdown("""
    <div class='k8s-node-box'>
        <div class='node-header'>
            <span>🖥️ storage-node-03</span>
            <span style='color: #f59e0b;'>Storage Tier</span>
        </div>
        <div style='color: #94a3b8; font-size: 11px; margin-bottom: 8px;'>Role: HDFS DataNode Primary Worker</div>
    """, unsafe_allow_html=True)
    
    for i in range(1, min(hdfs_count + 1, 4)):
        if is_hdfs_socket:
            st.markdown(f"<span class='pod-pill pod-red'>🔴 hdfs-datanode-0{i} (E4 DROPPED)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='pod-pill pod-green'>🟢 hdfs-datanode-0{i} (RUNNING)</span>", unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# Node 4: Storage Node-04 (Replica / Secondary)
with k_col4:
    is_cordoned = "storage-node-04" in st.session_state.cluster_state["cordoned_nodes"] or (last_res and "E7" in last_res.get("suspicious_events", []))
    box_class = "k8s-node-box-cordon" if is_cordoned else "k8s-node-box"
    
    st.markdown(f"""
    <div class='{box_class}'>
        <div class='node-header'>
            <span>🖥️ storage-node-04</span>
            <span style='color: {"#f97316" if is_cordoned else "#f59e0b"};'>{"⚠️ CORDONED" if is_cordoned else "Storage Replica"}</span>
        </div>
        <div style='color: #94a3b8; font-size: 11px; margin-bottom: 8px;'>Mount: /dev/sdb (Block Store)</div>
    """, unsafe_allow_html=True)
    
    if is_cordoned:
        st.markdown("<span class='pod-pill pod-yellow'>🟠 datanode-replica-04 (DRAINING)</span>", unsafe_allow_html=True)
        st.caption("🔒 Quarantined: Hardware Disk Failure Detected")
    elif hdfs_count > 3:
        st.markdown("<span class='pod-pill pod-green'>🟢 datanode-replica-01 (RUNNING)</span>", unsafe_allow_html=True)
        for i in range(4, hdfs_count + 1):
            st.markdown(f"<span class='pod-pill pod-blue'>🔵 hdfs-datanode-0{i} (NEW POD)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pod-pill pod-green'>🟢 datanode-replica-01 (RUNNING)</span>", unsafe_allow_html=True)
        st.markdown("<span class='pod-pill pod-green'>🟢 datanode-replica-02 (RUNNING)</span>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Human-In-The-Loop Approval Modal
# ---------------------------------------------------------------------------
if len(st.session_state.hitl_queue) > 0:
    st.warning("🚨 **HUMAN-IN-THE-LOOP SAFETY GUARDRAIL TRIGGERED**: High-Impact Action Intercepted.")
    
    for idx, item in enumerate(st.session_state.hitl_queue):
        with st.container():
            st.markdown(f"""
            **Requested Destructive Action:** `{item['task']}` on `{item['target_node']}`  
            **Reason:** {item['reason']} | **Risk Tier:** `HIGH` (Destructive Hardware Operation)  
            **Safety Rule:** Action cannot be auto-executed. Human administrator confirmation required.
            """)
            c1, c2, _ = st.columns([1.2, 1.2, 3])
            with c1:
                if st.button("✅ Authorize & Execute", key=f"app_{idx}", type="primary"):
                    st.success(f"Action '{item['task']}' authorized by Human Supervisor. Node cordoned and drained successfully.")
                    st.session_state.hitl_queue.pop(idx)
                    st.session_state.cluster_state["cluster_health"] = "MAINTENANCE IN PROGRESS (SAFE)"
                    time.sleep(1)
                    st.rerun()
            with c2:
                if st.button("❌ Deny & Abort", key=f"den_{idx}"):
                    st.info("Action rejected by administrator. Reverting node quarantine.")
                    st.session_state.hitl_queue.pop(idx)
                    st.session_state.cluster_state["cordoned_nodes"] = []
                    st.session_state.cluster_state["cluster_health"] = "HEALTHY"
                    time.sleep(1)
                    st.rerun()
    st.divider()


# ---------------------------------------------------------------------------
# Real-Time Multi-Agent Reasoning Pipeline
# ---------------------------------------------------------------------------
st.subheader("🧠 Multi-Agent Diagnostic Pipeline & Reasoning Trace")

if st.session_state.last_result:
    res = st.session_state.last_result
    
    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs(["⚡ 5-Agent Execution State", "🤖 Chain-of-Thought (SLM)", "📊 Human vs. AI SRE Impact", "📋 Audit Trail"])
    
    with t_tab1:
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            st.caption("1. Ingestion Agent")
            st.info(f"Source: **{res.get('log_source')}**\n\nSession: `{res.get('session_id')}`")
        with a_col2:
            st.caption("2. Perception Detection")
            if res.get("detection_status") == "ANOMALY":
                st.error(f"Status: **ANOMALY**\n\nConfidence: **{res.get('anomaly_confidence', 0)*100:.1f}%**")
            else:
                st.success(f"Status: **NORMAL**\n\nConfidence: **{(1 - res.get('anomaly_confidence', 0))*100:.1f}%**")
        with a_col3:
            st.caption("3. Risk Governance")
            risk = res.get("risk_tier", "NONE")
            if risk == "HIGH":
                st.error("Risk: **HIGH**\n\nAction: **Escalate to HITL**")
            elif risk == "LOW":
                st.warning("Risk: **LOW**\n\nAction: **Auto-Scale K8s**")
            else:
                st.success("Risk: **NONE**\n\nAction: **Nominal**")
        with a_col4:
            st.caption("4. Actuator Result")
            st.code(res.get("execution_status"), language="bash")

    with t_tab2:
        st.markdown("**Fine-Tuned Mistral-7B Chain-of-Thought (CoT) Output:**")
        cot_text = res.get("cot_reasoning", "Nominal operational state. Zero diagnostic reasoning required.").replace('\n', '<br>')
        st.markdown(f"<div class='cot-container'>{cot_text}</div>", unsafe_allow_html=True)
        
        st.markdown(f"**Root-Cause Diagnosis:** `{res.get('root_cause_diagnosis')}`")
        st.markdown(f"**Kubernetes Command:** `{json.dumps(res.get('action_details', {}))}`")

    with t_tab3:
        st.markdown("### 🏆 Benchmark Comparison: Human SRE vs. NIC-CloudBot Autonomous SRE")
        comp_data = {
            "Resolution Phase": ["1. Anomaly Detection", "2. Root-Cause Analysis (RCA)", "3. Scaling / Remediation Action", "Total Outage Duration"],
            "Traditional Human SRE (Industry Benchmark)": ["~15 minutes (Alert fatigue)", "~45 minutes (Manual grep / dashboard)", "~10 minutes (Manual kubectl apply)", "~70 minutes (Costly Downtime)"],
            "🤖 NIC-CloudBot (Our System)": ["0.18 ms (Random Forest Perception Tool)", "1.2 seconds (Fine-Tuned Mistral-7B SLM)", "3.8 seconds (Automated Actuator API)", "< 6 seconds (Auto-Recovered!)"]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    with t_tab4:
        st.write("Step-by-step state machine audit trail:")
        st.dataframe(pd.DataFrame(res.get("audit_trail", [])), use_container_width=True)
else:
    st.info("👈 Select an incident from the sidebar and click **'🚨 Inject Incident'** to begin the live demonstration!")


# ---------------------------------------------------------------------------
# Historical Audit Log Table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📜 Enterprise Audit Trail (Immutable Event Record)")
if len(st.session_state.audit_history) > 0:
    st.dataframe(pd.DataFrame(st.session_state.audit_history).iloc[::-1], use_container_width=True)
else:
    st.caption("No historical incidents logged in this session.")
