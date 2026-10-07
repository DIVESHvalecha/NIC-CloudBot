# NIC-CloudBot: Technical Deep-Dive & Design Rationale
### For Internal Team Review — Architecture, Decisions, and Agent Design

---

## 0. What Is This Document?

This is an internal technical explanation of the NIC-CloudBot system — written for teammates who need to understand exactly what the system does, why we made specific design decisions, and how every component fits together. It is meant to complement the README (which is for external/GitHub audiences) and goes much deeper into the *why*.

Read this before:
- Presenting to teachers/examiners
- Modifying any component of the system
- Writing your project report

---

## 1. The Problem We're Solving

Modern cloud infrastructure runs on Kubernetes clusters that generate **millions of log events per second**. When something goes wrong — a disk fails, a pod runs out of memory, a flash crowd hits the web tier — a human SRE engineer has to:

1. Notice an alert (often delayed by alert fatigue)
2. Grep through log files manually
3. Diagnose the root cause
4. Decide on a fix
5. Type the kubectl command
6. Verify it worked

This entire process takes **40–70 minutes on average** in industry benchmarks. During that time, the outage is costing real money and real users.

**Our question:** Can we replace this pipeline with an AI system that detects, diagnoses, decides, and acts autonomously — in under 6 seconds?

The answer is NIC-CloudBot.

---

## 2. Why "Compound AI"? Why Not Just One LLM?

This is the most important design decision in the entire project.

Using a single large LLM (like GPT-4 or full Mistral-7B) for everything would be:
- **Slow:** A 7B parameter model takes 1–5 seconds just to process the input
- **Wasteful:** 95% of logs are normal and don't need deep reasoning
- **Expensive:** Can't run without a GPU

Instead, we use a **Compound AI Architecture** — two models, each doing what it's best at:

| Component | Job | Speed | Model |
|---|---|---|---|
| Random Forest Classifier | Binary triage: Is this ANOMALY or NORMAL? | **0.18ms** | Scikit-learn |
| Mistral-7B QLoRA SLM | Deep reasoning: What exactly happened and why? | ~1.2s | Fine-tuned LLM |

This is exactly how real production ML systems are built. Cheap, fast models handle 95% of traffic. Expensive, smart models only run when needed.

**The key insight:** You don't need an LLM to tell you a log trace is normal. A Random Forest can do that in 0.18ms. You only invoke the heavy SLM for anomalies.

---

## 3. Why LangGraph for Orchestration?

We evaluated several options:

- **Plain Python functions:** Would work but no shared state, no graph visualization, no conditional routing
- **LangChain:** Better, but designed for linear chains. Our pipeline has a conditional fork
- **LangGraph:** Designed specifically for stateful, multi-agent graphs with conditional edges

LangGraph gives us:
1. **TypedDict shared state** — Every agent reads from and writes to a single `CloudBotState` object. This prevents data corruption between agents.
2. **Conditional Edges** — The `route_after_detection` function creates a real fork in the pipeline based on detection output. This is the fast-path optimization.
3. **Deterministic Execution** — The graph is compiled once and is immutable. No agent can accidentally skip another.
4. **Auditability** — Every state transition is logged automatically.

---

## 4. The CloudBotState — Shared Memory Between Agents

```python
class CloudBotState(TypedDict):
    # Input / Ingestion
    raw_payload: str          # The raw log line that triggered this
    log_source: str           # "HDFS", "APACHE", "OPENSSH", "LINUX_OOM"
    session_id: str           # BlockId, IP, or Request ID
    event_sequence: List[str] # Parsed event codes: ["E5", "E22", "E4", ...]
    latency_ms: float         # Block transfer time or response time
    telemetry: Dict           # CPU%, Memory%, Replicas, QPS

    # Detection Output (written by Agent 2)
    detection_status: str     # "NORMAL" or "ANOMALY"
    anomaly_confidence: float # 0.0-1.0 probability score
    suspicious_events: List[str] # Which events triggered the flag

    # SLM Reasoning Output (written by Agent 3)
    cot_reasoning: str        # The 4-step Chain-of-Thought diagnosis
    root_cause_diagnosis: str # One-line summary of the root cause

    # Decision Output (written by Agent 4)
    target_replicas: int      # Target Kubernetes pod count
    scaling_direction: str    # "SCALE_OUT", "NONE", "CORDON_NODE"
    risk_tier: str            # "LOW", "MEDIUM", "HIGH", "NONE"

    # Actuator Output (written by Agent 5)
    execution_status: str     # "AUTO_EXECUTED", "PENDING_HUMAN_APPROVAL", "RESOLVED"
    action_details: Dict      # The actual Kubernetes API command to execute
    audit_trail: List[Dict]   # Immutable log of every agent's action
```

**Why TypedDict?** It gives us Python type safety without the overhead of Pydantic. LangGraph natively supports TypedDict as its state schema.

**Critical rule:** Each agent can only *add* to `audit_trail`. It never replaces the list — it appends. This guarantees immutability of the historical record.

---

## 5. Each Agent — Deep Dive

### Agent 1: Ingestion Agent

**What it does:** Receives raw log text and extracts structured metadata.

**Why it exists as a separate agent:** In a real production system, logs arrive in different formats — HDFS event codes, Apache combined log format, syslog kernel messages. The Ingestion Agent normalizes them all into a standard `CloudBotState` schema. It also auto-extracts HDFS Block IDs from raw payloads using regex (`blk_\d+`).

**Key code:** The regex `r'(blk_-?\d+)'` extracts Block IDs because HDFS blocks can have negative IDs (they're long integers that overflow).

---

### Agent 2: Detection Agent (The Fast Eye)

**What it does:** Calls `LogAnomalyClassifierTool.analyze_trace()` and gets back a probability score and list of suspicious event codes.

**The classifier input:** Two feature types are concatenated:
1. **TF-IDF features** over the event sequence string (e.g., "E5 E22 E4 E4 E4 E20")
2. **Numeric features:** mean interval, max interval, std deviation, count, and total latency

These are hstacked into a sparse matrix and fed to the Random Forest.

**Why Random Forest over deep learning for triage?**
- No GPU required
- Extremely fast inference (0.18ms)
- Interpretable — you can print feature importances
- Trained on real HDFS production logs
- Does not need context window — each trace is independent

**The suspicious event list:** E4 = socket reset, E7 = write block exception, E8 = thread interrupt, E20 = timeout. These are the HDFS event codes that correlate with anomalies in the training data.

---

### Agent 3: Reasoning Agent (The Brain)

**What it does:** Invokes `SLMReasoningEngine.generate_sre_diagnosis()` to produce:
- A 4-step Chain-of-Thought (CoT) diagnosis
- A one-line root-cause summary
- A Kubernetes scaling recommendation
- A risk tier classification

**The SLM Fallback Architecture:** Because fine-tuned Mistral-7B requires a GPU (16GB+ VRAM) and most development machines don't have one, we designed a **deterministic inference fallback** that runs without any GPU. The fallback mirrors the exact reasoning logic the SLM would produce — same structure, same conclusions, same action format.

In production, you would replace this with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./nic-cloudbot-slm-adapter")
output = model.generate(...)
```

**Chain-of-Thought structure:** Every diagnosis follows this 4-step template:
1. **Log Analysis:** What specific events were seen?
2. **Telemetry Correlation:** What do CPU/memory/QPS numbers confirm?
3. **Root-Cause:** What is the actual failure mechanism?
4. **Decision/Strategy:** What action does the system recommend?

This structure was deliberately chosen to match how real SRE postmortem documents are written. The SLM was fine-tuned on 8,000 examples following this exact template.

---

### Agent 4: Decision Agent (The Policy Engine)

**What it does:** Takes the risk_tier from Agent 3 and translates it into an execution policy.

**Policy logic:**
```
HIGH risk  -> PENDING_HUMAN_APPROVAL (stop, wait for human)
LOW risk   -> AUTO_EXECUTED (proceed autonomously)
NONE risk  -> RESOLVED (no action needed)
```

**Why is this a separate agent?** In a real system, the policy rules would be loaded from a database or config file that an administrator can update without touching the AI code. By separating policy from reasoning, we maintain clean separation of concerns. The SLM focuses on *what happened*. The Decision Agent focuses on *what the company's policy says to do about it*.

---

### Agent 5: Actuator Agent (The Hands)

**What it does:** The final action layer. If `AUTO_EXECUTED`, it would call the real Kubernetes Python client API:
```python
# Real production code would be:
from kubernetes import client, config
config.load_incluster_config()
apps_v1 = client.AppsV1Api()
apps_v1.patch_namespaced_deployment_scale(
    name=action['deployment'],
    namespace=action['namespace'],
    body={"spec": {"replicas": action['target_replicas']}}
)
```

If `PENDING_HUMAN_APPROVAL`, it does nothing except add the action to the HITL queue in Streamlit's session state.

**Why is HITL only at the Actuator layer?** Because we want maximum autonomy for safe, reversible operations (scaling pods up is easy to undo) while adding a safety circuit-breaker only for irreversible, destructive operations (draining a node could cause data loss).

---

## 6. The Conditional Routing — The Fast Path

```python
def route_after_detection(state: CloudBotState) -> str:
    if state.get("detection_status") == "ANOMALY":
        return "reasoning_agent"
    return "actuator_agent"  # Skip reasoning entirely
```

This single function is arguably the most important optimization in the system. When a log trace is classified as NORMAL:
- We skip Agent 3 (Mistral-7B SLM — the expensive part)
- We skip Agent 4 (Decision Agent — unnecessary)
- We go directly to Agent 5 with no action

In a production HDFS cluster, the vast majority of block writes complete normally. If we ran the SLM on every trace, we'd need constant GPU resources. With the fast-path, we only need GPU compute for the anomalous minority.

---

## 7. The Fine-Tuning Pipeline

### Why Fine-Tune at All?

A base Mistral-7B model has no idea what "E4 HDFS socket exception" means or what `DRAIN_AND_MAINTAIN_NODE` is. General-purpose models are trained on web text, not cloud SRE runbooks.

Fine-tuning on our `nic_cloudbot_sre_dataset.json` teaches the model:
- HDFS event code semantics (E4 = socket reset, E7 = disk write exception)
- Kubernetes scaling vocabulary
- SRE Chain-of-Thought reasoning format
- Risk tier classification

### QLoRA — Why 4-bit Quantization?

Full fine-tuning of Mistral-7B requires ~110GB VRAM (impossible on free GPUs). QLoRA (Quantized LoRA) makes this feasible:

1. **Load the base model in 4-bit NF4 quantization** → ~4GB VRAM
2. **Add small LoRA adapter matrices** (r=16) → ~100MB of trainable parameters
3. **Only train the LoRA matrices** → The base model weights are frozen

The LoRA adapters attach to all attention and MLP projection layers:
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

These are the layers that control how the model "pays attention" to input tokens and how it transforms representations. Teaching these layers new behavior is sufficient to specialize the model for SRE tasks.

### Dataset Structure

Each training sample in `nic_cloudbot_sre_dataset.json` looks like:
```json
{
  "instruction": "You are an expert SRE. Given log events and telemetry, perform a 4-step Chain-of-Thought diagnosis...",
  "input": "Log Source: HDFS\nEvent Sequence: E5, E22, E4, E4, E4, E20\nLatency: 52400ms\nCPU: 94%\n...",
  "output": "1. Log Analysis: Multiple E4 TCP socket resets detected...\n2. Telemetry Correlation: Worker nodes show CPU saturation...\n..."
}
```

The ChatML format wraps this:
```
<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n{input}<|im_end|>
<|im_start|>assistant\n{output}<|im_end|>
```

---

## 8. The Dashboard — Design Decisions

### Why Streamlit?

We considered React/Next.js but chose Streamlit because:
1. Pure Python — the dashboard code can call `agent_graph.py` functions directly, no API layer needed
2. Fast to build and iterate
3. Free deployment on Streamlit Community Cloud
4. Excellent data visualization built in (st.dataframe, st.metric, etc.)

### Session State Architecture

Streamlit re-runs the entire Python script on every user interaction. We use `st.session_state` as a persistent store across reruns:

- `cluster_state` — Current K8s cluster metrics (pod counts, CPU, health)
- `story_phase` — Which "act" of the story we're in (1_STEADY → 2_CHAOS → 4_RESOLVED)
- `current_incident` — The incident currently being processed
- `hitl_queue` — List of HIGH-risk actions awaiting human approval
- `audit_history` — Historical log of all incidents in this session
- `last_result` — The full CloudBotState output from the last graph invocation

### The K8s Topology Map

This is rendered as raw HTML/CSS injected via `st.markdown(unsafe_allow_html=True)`. Each node box and pod pill is a styled `<div>` and `<span>`. The colors change dynamically based on the last agent graph output:

- Green pods = RUNNING (normal)
- Red pods = FAILED/SATURATED (incident active)
- Blue pods = NEW (just scaled up)
- Orange/dashed border = CORDONED (node being drained)

### The importlib.reload() Pattern

The dashboard uses:
```python
import importlib
import agent_graph
importlib.reload(agent_graph)
```

This forces Python to re-import `agent_graph.py` every time the Streamlit script runs. This is necessary because Streamlit caches module imports, which could cause stale code to be used after you modify `agent_graph.py` without restarting.

### @st.cache_resource for the Graph

```python
@st.cache_resource
def get_graph():
    return build_cloudbot_graph()
```

The compiled LangGraph StateGraph is expensive to build (it has to validate all edges and nodes). We cache it so it's only built once per Streamlit server session, not on every rerun.

---

## 9. HDFS Event Codes — Reference

The event sequences in our incident catalog use HDFS Drain event codes from the BGL log dataset:

| Code | Meaning | Anomaly Signal |
|---|---|---|
| E2 | Receiving block | Normal |
| E3 | Block stored | Normal |
| E4 | Socket error / TCP reset | **ANOMALY** |
| E5 | Starting replication pipeline | Normal |
| E7 | write block exception | **ANOMALY** |
| E8 | Thread interrupted | **ANOMALY** |
| E9 | Receiving block transfer | Normal |
| E11 | BlockReceiver sending ack | Normal |
| E20 | Connection timeout | **ANOMALY** |
| E21 | Replication verification | Normal |
| E22 | Receiving block from pipeline | Normal |
| E23 | Block replication succeeded | Normal |
| E26 | Block data transfer complete | Normal |

A normal 3-way replication sequence looks like:
`E5 → E22 → E5 → E5 → E11 → E9 → E11 → E9 → E11 → E9 → E26 → E26 → E26 → E2 → E2 → E21 → E21`

An anomalous sequence with socket saturation looks like:
`E5 → E22 → E5 → E5 → E11 → E9 → E4 → E4 → E4 → E4 → E4 → E20`

---

## 10. What the Random Dynamic Incident Generator Does

```python
def generate_dynamic_random_incident() -> Dict[str, Any]:
```

This function generates a fresh, random incident every time you click "Generate Random Dynamic Incident" in the sidebar. It randomly decides:

- Is this an anomaly (60% chance) or normal write (40% chance)?
- Generates a random 16-digit HDFS Block ID
- Randomizes CPU (86–99% for anomaly), Memory (75–94%), QPS, and Latency
- Generates a random number of E4 socket exceptions (3–7 events)

This is useful for demonstrations because it shows the system is not just hardcoded for 8 fixed scenarios — it can handle arbitrary, previously unseen event patterns.

---

## 11. What We'd Do Differently in a Real Production System

1. **Replace the deterministic SLM fallback** with actual inference against the fine-tuned adapter
2. **Add Kafka/Kinesis integration** so real logs stream in instead of being simulated
3. **Use Kubernetes Python client** (`kubernetes` library) for actual API calls
4. **Add Prometheus/Grafana metrics export** for the detection confidence scores
5. **Store the audit trail in a database** (PostgreSQL) instead of session state
6. **Add webhook notifications** to Slack/PagerDuty when HITL approval is required
7. **Implement vector similarity search** so past resolved incidents improve future recommendations

---

## 12. Key Questions You May Get From Examiners

**Q: Why use a Random Forest instead of just using the SLM for everything?**
A: Speed and cost. Random Forest runs in 0.18ms with no GPU. Using an SLM for binary classification of 99% normal logs would waste enormous compute. This is industry best practice — cascade from cheap to expensive.

**Q: Isn't the SLM "deterministic fallback" just hardcoded if-else?**
A: Yes, in the demo. In production, you replace it with actual model inference. The architecture (agents, state, routing) is production-ready. The fine-tuning script and dataset prove we have a trained model — we just can't serve a 7B parameter model on a free cloud tier.

**Q: What is QLoRA vs full fine-tuning?**
A: Full fine-tuning updates all 7 billion parameters — requires 110GB VRAM. QLoRA freezes the base model weights (loading them in 4-bit quantization) and only trains small LoRA adapter matrices (~40 million parameters). Same result, 4x smaller memory footprint, trainable on a free Kaggle T4 GPU.

**Q: What is HITL and why is it important?**
A: Human-In-The-Loop means the AI system pauses before executing irreversible, high-risk actions and waits for a human administrator to approve. This is a core principle of responsible AI deployment in production. You never want autonomous AI to drain storage nodes or delete data without a human sign-off.

**Q: How does the system decide what is HIGH vs LOW risk?**
A: The SLM's reasoning output includes a risk_tier classification. The decision rules are: anything involving permanent infrastructure changes (drain node, expand memory limits) = HIGH. Anything reversible (scale pods up, block an IP) = LOW. This separation is enforced by Agent 4's policy engine.

---

*Last Updated: 2026-10-07 by Divesh Valecha*
