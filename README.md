# 🛡️ NIC-CloudBot: Autonomous Cloud SRE Operations System

> **A Compound AI Architecture for Autonomous Cloud Incident Response using 5-Agent LangGraph Orchestration, Fine-Tuned Mistral-7B SLM, and High-Performance ML Perception.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**NIC-CloudBot** is a production-grade, autonomous Site Reliability Engineering (SRE) system built as a **Compound AI architecture**. It combines two complementary AI systems:

1. **Fast Perception Layer** — A Random Forest classifier for sub-millisecond log anomaly triage (<0.2ms inference).
2. **Deep Reasoning Layer** — A QLoRA fine-tuned Mistral-7B Small Language Model (SLM) for expert-level root-cause diagnosis and remediation planning.

These two models are orchestrated by a **5-agent LangGraph StateGraph** that mirrors how a real cloud SRE team operates: detect, analyze, decide, and act — all with a Human-In-The-Loop (HITL) safety guardrail for high-risk operations.

The system targets real production cloud incidents including **HDFS storage failures**, **Kubernetes pod scaling**, **SSH brute-force attacks**, **OOM kernel events**, and **web traffic surge handling**.

---

## System Architecture

```
                        +-----------------------------------------------+
                        |         LIVE LOG INGESTION PIPELINE            |
                        |  (HDFS Event Traces, Apache Logs, Syslog)      |
                        +-------------------+---------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------+
|                    LANGGRAPH STATEFUL ORCHESTRATION                         |
|                                                                             |
|   +-----------+   +-----------+   +-------------------+                    |
|   | Agent 1   |-->| Agent 2   |-->|  Conditional      |                    |
|   | INGESTION |   | DETECTION |   |    ROUTER         |                    |
|   |           |   | (RF <0.2ms)   |                   |                    |
|   | Sessionize|   | Classify &|   | ANOMALY ------+   |                    |
|   | Parse Logs|   | Score Logs|   | NORMAL  --+   |   |                    |
|   +-----------+   +-----------+   +-----------+---+---+                    |
|                                               |   |                        |
|   +------------------+   +------------------++   |                        |
|   |    Agent 3       |<--|                  ||<--+                         |
|   |   REASONING      |   |   Agent 4        |                             |
|   |  (Mistral-7B     |-->|   DECISION       |                             |
|   |   QLoRA SLM)     |   | (Policy Engine)  |                             |
|   |  Chain-of-Thought|   | LOW/HIGH Risk    |                             |
|   +------------------+   +------------------+                             |
|                                     |                                      |
|                                     v                                      |
|                          +------------------+                             |
|                          |    Agent 5       |                             |
|                          |    ACTUATOR      |                             |
|                          | (HITL Guardrail) |                             |
|                          |                  |                             |
|                          | LOW  --> k8s API |                             |
|                          | HIGH --> Human   |                             |
|                          |        Approval  |                             |
|                          +------------------+                             |
|                                                                             |
|   CloudBotState (TypedDict) <-- Shared, Immutable State Across Agents      |
+-----------------------------------------------------------------------------+
                                            |
                                            v
                        +-----------------------------------------------+
                        |     STREAMLIT OPERATIONS CENTER DASHBOARD     |
                        |  * Live K8s Topology Map                      |
                        |  * SLM Chain-of-Thought Viewer                |
                        |  * HITL Admin Approval Queue                  |
                        |  * Enterprise Audit Trail (Immutable)         |
                        +-----------------------------------------------+
```

---

## 🤖 The 5 Agents — What Each One Does

| # | Agent | Role | Speed |
|---|-------|------|-------|
| 1 | **Ingestion Agent** | Log sessionization & parsing. Extracts Block IDs, session context, and event sequences from raw log payloads. | ~1ms |
| 2 | **Detection Agent** | Runs the fast Random Forest Perception Classifier on the event sequence. Returns ANOMALY/NORMAL + confidence score. | **<0.2ms** |
| 3 | **Reasoning Agent** | Fine-Tuned Mistral-7B SLM performs Chain-of-Thought (CoT) root-cause diagnosis. Identifies the exact failure mechanism and proposes a remediation plan. | ~1.2s |
| 4 | **Decision Agent** | Cloud SRE Policy Engine. Evaluates the risk tier (LOW/HIGH/NONE) and sets execution policy: `AUTO_EXECUTED` or `PENDING_HUMAN_APPROVAL`. | ~1ms |
| 5 | **Actuator Agent** | Executes cloud API calls (Kubernetes scale, firewall block) OR pauses and routes HIGH-risk destructive operations to the HITL admin queue. | ~2ms |

### Fast-Path Optimization

When Agent 2 classifies a log as `NORMAL`, the pipeline **skips Agent 3 entirely** (bypasses the heavy SLM), routing directly to Agent 5. This conditional edge saves significant GPU compute on healthy traffic — a critical production optimization.

---

## 🔬 Model Architecture — Compound AI

### Layer 1: Fast Perception Classifier
- **Model:** Scikit-learn Random Forest
- **Input:** TF-IDF vectorized HDFS event sequences + numeric telemetry features (latency, CPU, memory)
- **Training Data:** HDFS log event traces (BGL Dataset structure), 100k+ samples
- **Inference:** **0.18ms** — fast enough for real-time stream processing
- **Purpose:** Binary triage (ANOMALY / NORMAL). Acts as the "first eye" of the system.

### Layer 2: Fine-Tuned SLM (Reasoning Brain)
- **Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Fine-Tuning Method:** QLoRA (Quantized Low-Rank Adaptation) — 4-bit NF4 quantization
- **Training Data:** `nic_cloudbot_sre_dataset.json` — a custom 8,000-sample instruction-tuning dataset with SRE Chain-of-Thought reasoning traces
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **LoRA Config:** r=16, alpha=32
- **Purpose:** Expert-level root-cause analysis, producing structured CoT diagnostic reports and Kubernetes scaling recommendations
- **Memory:** 4-bit quantization reduces VRAM from ~28GB (FP32) to ~7GB (INT4) — **4x compression**

---

## 📊 Performance Benchmarks

### NIC-CloudBot vs. Traditional Human SRE Team

| Resolution Phase | Human SRE (Industry Average) | NIC-CloudBot |
|---|---|---|
| Anomaly Detection | ~15 minutes (alert fatigue) | **0.18 ms** (RF Classifier) |
| Root-Cause Analysis | ~45 minutes (manual grep) | **1.2 seconds** (Mistral-7B SLM) |
| Remediation Execution | ~10 minutes (manual kubectl) | **3.8 seconds** (Actuator API) |
| **Total Outage Duration** | **~70 minutes** | **< 6 seconds** |

### Quantization Memory Matrix

| Precision | Model Size | VRAM Required | Inference Speed | Quality Loss |
|---|---|---|---|---|
| FP32 (Full) | 28.0 GB | 32 GB+ | Baseline | None |
| FP16 (Half) | 14.0 GB | 16 GB | ~1.85x faster | Negligible |
| INT8 | 7.5 GB | 10 GB | ~2.8x faster | Minimal |
| **INT4 / NF4 (Ours)** | **4.2 GB** | **6-8 GB** | **~3.9x faster** | **Acceptable** |

---

## 🎭 Incident Catalog (8 Real-World Scenarios)

The system includes a built-in chaos engineering catalog for demonstration and testing:

| # | Scenario | Risk | Expected Action |
|---|----------|------|----------------|
| 1 | Normal 3-Way Block Write | NONE | Zero action — system healthy |
| 2 | DataNode Socket Saturation (E4 exceptions) | LOW | Auto-scale HDFS DataNode pods (+2) |
| 3 | Storage Hardware Degradation (E7 disk failure) | **HIGH** | HITL: Drain & cordon node-04 |
| 4 | Web Flash Crowd (Apache HTTP 504) | LOW | Auto-scale web-frontend pods (+4) |
| 5 | SSH Brute Force Attack | LOW | Auto-block attacker IP via VPC rule |
| 6 | Linux OOM Killer Activation | **HIGH** | HITL: Expand pod memory limit (2Gi to 4Gi) |
| 7 | Broken Replication Pipeline Cutoff | LOW | Auto-schedule block re-replication |
| 8 | Cascading I/O Bottleneck Spike | LOW | Scale-out DataNode workers |

A **random dynamic incident generator** can also produce authentic, unique incidents with randomized telemetry (CPU, memory, QPS, block IDs) for live demos.

---

## 🗂️ Project Structure

```
major-project/
├── agent_graph.py              # Core: 5-agent LangGraph StateGraph
├── dashboard.py                # Streamlit Operations Center UI
├── classifier_tool.py          # Fast RF Inference Tool (loaded by Agent 2)
├── train_classifier.py         # Training script for Random Forest classifier
├── train_slm_qlora.py          # QLoRA fine-tuning script (runs on Kaggle/Colab)
├── generate_slm_dataset.py     # SRE CoT dataset generation pipeline
├── evaluate_deduplicated.py    # Model evaluation & deduplication script
├── quantize_benchmark.py       # INT4 quantization performance benchmarking
├── quantization_matrix.md      # Quantization benchmark results
├── test_classifier.py          # Unit tests for the ML classifier
│
├── log_anomaly_classifier.pkl  # [Trained] RF Classifier Model (gitignored)
├── event_vectorizer.pkl        # [Trained] TF-IDF Event Vectorizer (gitignored)
├── Event_traces.csv            # [Data] Raw HDFS event traces (gitignored)
├── HDFS_templates.csv          # [Data] HDFS log templates reference
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions (models, data, venv)
├── explanation.md              # Deep-dive technical documentation for team
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/nic-cloudbot-sre.git
cd nic-cloudbot-sre
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `torch`, `transformers`, `peft`, `trl`, and `bitsandbytes` are only required for SLM fine-tuning. The dashboard runs without GPU using the deterministic SLM fallback.

### 3. Launch the Dashboard
```bash
streamlit run dashboard.py
```

The Operations Center will open at `http://localhost:8501`.

### 4. Run a Demo Incident
1. Open the sidebar — Select Incident Scenario
2. Click **Inject Incident**
3. Watch the 5-agent pipeline execute in real-time with visual K8s topology updates

---

## 🧪 Training Your Own Models

### Train the Random Forest Classifier
```bash
python train_classifier.py
```
Requires `Event_traces.csv` and `anomaly_label.csv`. Outputs `log_anomaly_classifier.pkl` and `event_vectorizer.pkl`.

### Generate the SRE Fine-Tuning Dataset
```bash
python generate_slm_dataset.py
```
Generates `nic_cloudbot_sre_dataset.json` with CoT reasoning traces.

### Fine-Tune Mistral-7B with QLoRA (GPU Required)
Run on Kaggle (Free T4 GPU) or Google Colab (Free A100):
```bash
python train_slm_qlora.py
```
Outputs adapter weights to `./nic-cloudbot-slm-adapter/`. Upload dataset to Kaggle first.

---

## 🌐 Deployment (Free Options)

### Option A: Streamlit Community Cloud (Recommended)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set **Main file** to `dashboard.py`
4. Deploy — gets a free public URL at `https://your-app.streamlit.app`

### Option B: Hugging Face Spaces
1. Create a new Space, select **Streamlit** SDK
2. Push code + `requirements.txt` via git
3. Free, always-on URL at `https://huggingface.co/spaces/<user>/<space>`

---

## 🛡️ Human-In-The-Loop (HITL) Safety Architecture

NIC-CloudBot implements a **two-tier risk classification** to prevent autonomous AI from executing destructive operations without human oversight:

| Risk Tier | Example Actions | Execution Policy |
|---|---|---|
| LOW | Scale K8s pods, block firewall IP | AUTO_EXECUTED immediately |
| HIGH | Drain K8s node, expand pod limits | PAUSED → HITL Admin Queue |
| NONE | Healthy system, no changes needed | RESOLVED (no action) |

All actions are logged to an **immutable enterprise audit trail** with timestamps, agent IDs, and decision rationale.

---

## 📜 License

MIT License

---

## 👥 Authors

- Divesh Valecha, Palak Chainani,Jeet Suchdev, Meet RamRakhiani.

---

*Built as a Major Project demonstrating advanced AI/ML concepts: multi-agent orchestration, compound AI systems, PEFT fine-tuning, and autonomous cloud operations.*
