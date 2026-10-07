import json
import random
import pandas as pd
import numpy as np
import time

print("=" * 70)
print("  NIC-CloudBot: SRE & Cloud Scaling Dataset Generator for SLM Tuning")
print("=" * 70)

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Load Templates and Traces
print("\n[1/4] Loading HDFS Templates and Event Traces...")
df_templates = pd.read_csv('HDFS_templates.csv')
template_map = dict(zip(df_templates['EventId'], df_templates['EventTemplate']))

df_traces = pd.read_csv('Event_traces.csv')
df_labels = pd.read_csv('anomaly_label.csv')
df = pd.merge(df_traces, df_labels, on='BlockId')
df['target'] = (df['Label_y'] == 'Anomaly').astype(int)

# Separate normal and anomaly traces
df_normal = df[df['target'] == 0]
df_anomaly = df[df['target'] == 1]

print(f"    Available Normal Traces : {len(df_normal):,}")
print(f"    Available Anomaly Traces: {len(df_anomaly):,}")

dataset = []

SYSTEM_PROMPT = (
    "You are NIC-CloudBot, an enterprise autonomous Site Reliability Engineering (SRE) "
    "and Cloud Resource Scaling Agent for private and hybrid cloud infrastructure. "
    "Analyze the incoming server incident alert, execute Chain-of-Thought (CoT) diagnosis, "
    "identify the root cause, and determine the exact remediation and cloud scaling action."
)

# 2. Generator Functions for Different Incident Types

def generate_hdfs_normal(row):
    events = [e.strip() for e in row['Features'].strip("[]").split(",") if e.strip()]
    latency = row['Latency']
    block_id = row['BlockId']
    
    cpu = random.randint(15, 45)
    mem = random.randint(25, 50)
    replicas = random.randint(3, 5)
    
    input_text = (
        f"Service: HDFS Storage Cluster\n"
        f"BlockId: {block_id}\n"
        f"Classifier Output: NORMAL (Confidence: {random.uniform(0.01, 0.20):.4f})\n"
        f"Event Sequence: [{', '.join(events[:15])}{'...' if len(events) > 15 else ''}]\n"
        f"Total Events: {len(events)} | Latency: {latency}ms\n"
        f"Cluster Telemetry: CPU={cpu}%, Memory={mem}%, Active Pod Replicas={replicas}"
    )
    
    output_text = (
        f"### Chain-of-Thought (CoT) Reasoning:\n"
        f"1. Log Analysis: The event sequence exhibits standard 3-way pipeline replication (E5 allocation ➔ E11/E9 transfer ➔ E26/E21 verification).\n"
        f"2. Telemetry Check: DataNode CPU ({cpu}%) and Memory ({mem}%) are well within nominal operational limits.\n"
        f"3. Latency Evaluation: Operation latency ({latency}ms) is healthy for distributed block commit.\n"
        f"4. Decision: No service degradation or bottlenecks detected. System is fully operational.\n\n"
        f"### Diagnosis:\n"
        f"Status: HEALTHY. Block replication completed successfully across all replica nodes with zero exceptions.\n\n"
        f"### Remediation Action:\n"
        f"- Action: log_cluster_health(status='HEALTHY', block_id='{block_id}')\n"
        f"- Cloud Scaling: NONE_REQUIRED\n"
        f"- Risk Level: NONE"
    )
    
    return {"instruction": SYSTEM_PROMPT, "input": input_text, "output": output_text, "category": "HDFS_NORMAL"}

def generate_hdfs_network_anomaly(row):
    events = [e.strip() for e in row['Features'].strip("[]").split(",") if e.strip()]
    latency = max(row['Latency'], random.randint(35000, 60000))
    block_id = row['BlockId']
    
    cpu = random.randint(88, 98)
    mem = random.randint(75, 92)
    cur_replicas = random.randint(3, 4)
    target_replicas = cur_replicas + random.randint(2, 3)
    
    e4_count = events.count('E4')
    
    input_text = (
        f"Service: HDFS Storage Cluster\n"
        f"BlockId: {block_id}\n"
        f"Classifier Output: ANOMALY (Confidence: {random.uniform(0.85, 0.99):.4f})\n"
        f"Flagged Anomalous Events: ['E4' (Got exception while serving block), 'E20']\n"
        f"Event Sequence: [{', '.join(events[:20])}{'...' if len(events) > 20 else ''}]\n"
        f"Total Events: {len(events)} | Latency: {latency}ms\n"
        f"Cluster Telemetry: CPU={cpu}% (HIGH), Memory={mem}%, Active Pod Replicas={cur_replicas}"
    )
    
    output_text = (
        f"### Chain-of-Thought (CoT) Reasoning:\n"
        f"1. Log Analysis: Multiple E4 exceptions detected ({e4_count} occurrences) along with packet responder timeouts (E20). This indicates socket connection resets while serving block streams.\n"
        f"2. Telemetry Correlation: DataNode worker nodes are experiencing extreme CPU saturation ({cpu}%) and high I/O latency ({latency}ms).\n"
        f"3. Root-Cause: DataNode worker pods are overwhelmed by concurrent read/write requests, leading to thread pool exhaustion and dropped socket connections.\n"
        f"4. Scaling Strategy: Horizontal scaling of the DataNode worker deployment will distribute I/O load across more nodes and relieve packet contention.\n\n"
        f"### Diagnosis:\n"
        f"Status: ANOMALY - CRITICAL. Network packet responder thread starvation and socket resets due to DataNode pod saturation.\n\n"
        f"### Remediation Action:\n"
        f"- Action: scale_kubernetes_deployment(deployment='hdfs-datanode', current_replicas={cur_replicas}, target_replicas={target_replicas}, namespace='storage')\n"
        f"- Cloud Scaling: SCALE_OUT (+{target_replicas - cur_replicas} Pods)\n"
        f"- Risk Level: LOW (Auto-Executed via Horizontal Pod Autoscaler)"
    )
    
    return {"instruction": SYSTEM_PROMPT, "input": input_text, "output": output_text, "category": "HDFS_NETWORK_ANOMALY"}

def generate_hdfs_disk_anomaly(row):
    events = [e.strip() for e in row['Features'].strip("[]").split(",") if e.strip()]
    block_id = row['BlockId']
    node_id = f"node-{random.randint(1, 12):02d}"
    
    input_text = (
        f"Service: HDFS Storage Cluster\n"
        f"BlockId: {block_id}\n"
        f"Classifier Output: ANOMALY (Confidence: {random.uniform(0.92, 0.99):.4f})\n"
        f"Flagged Anomalous Events: ['E7' (writeBlock received exception), 'E8' (PacketResponder Interrupted)]\n"
        f"Event Sequence: [{', '.join(events[:15])}]\n"
        f"Cluster Telemetry: Target Node={node_id}, I/O Error Count=24, Disk Latency=Spike"
    )
    
    output_text = (
        f"### Chain-of-Thought (CoT) Reasoning:\n"
        f"1. Log Analysis: Detected severe block write exceptions (E7) followed by immediate packet responder interruption (E8) on {node_id}.\n"
        f"2. Telemetry Correlation: Physical disk I/O errors and write stall signals confirm underlying block device / filesystem failure.\n"
        f"3. Risk Evaluation: Draining or restarting an active storage node carries high risk of data replication storm if executed without supervisor oversight.\n"
        f"4. Decision: Isolate {node_id} from write allocation and route destructive maintenance request to Human-In-The-Loop approval queue.\n\n"
        f"### Diagnosis:\n"
        f"Status: ANOMALY - HARDWARE FAILURE. Probable physical disk degradation / bad sector write failure on DataNode {node_id}.\n\n"
        f"### Remediation Action:\n"
        f"- Action: request_human_approval(action='DRAIN_AND_MAINTAIN_NODE', target_node='{node_id}', reason='Hardware disk write failure detected on {block_id}')\n"
        f"- Cloud Scaling: CORDON_NODE_AND_REROUTE\n"
        f"- Risk Level: HIGH (Requires Human Administrator Authorization)"
    )
    
    return {"instruction": SYSTEM_PROMPT, "input": input_text, "output": output_text, "category": "HDFS_DISK_ANOMALY"}

def generate_web_ingress_surge():
    qps = random.randint(4500, 12000)
    err_rate = round(random.uniform(12.5, 34.0), 2)
    cur_replicas = random.randint(3, 6)
    target_replicas = cur_replicas + random.randint(4, 7)
    
    input_text = (
        f"Service: Apache / Nginx Ingress Controller\n"
        f"Alert Type: INGRESS_GATEWAY_TIMEOUT\n"
        f"Classifier Output: ANOMALY (Confidence: 0.9820)\n"
        f"Log Pattern: ['HTTP 504 Gateway Timeout', 'upstream timed out (110: Connection timed out)', 'client request queue full']\n"
        f"Cluster Telemetry: Traffic QPS={qps}, 5xx Error Rate={err_rate}%, Ingress CPU=94%, Active Pod Replicas={cur_replicas}"
    )
    
    output_text = (
        f"### Chain-of-Thought (CoT) Reasoning:\n"
        f"1. Log Analysis: Spike in HTTP 504 Gateway Timeouts with upstream backend connection drops.\n"
        f"2. Telemetry Correlation: Incoming query load ({qps} QPS) has exceeded the capacity of {cur_replicas} active web pods, causing 5xx error rate to surge to {err_rate}%.\n"
        f"3. Root-Cause: Sudden flash-crowd traffic surge causing HTTP request backlog in the Nginx reverse proxy buffers.\n"
        f"4. Scaling Strategy: Immediately trigger autoscaling to add {target_replicas - cur_replicas} additional web pods to restore response times under 200ms.\n\n"
        f"### Diagnosis:\n"
        f"Status: ANOMALY - CAPACITY EXHAUSTION. Frontend web ingress tier overloaded by flash-crowd traffic spike.\n\n"
        f"### Remediation Action:\n"
        f"- Action: scale_kubernetes_deployment(deployment='web-frontend-service', current_replicas={cur_replicas}, target_replicas={target_replicas}, namespace='production')\n"
        f"- Cloud Scaling: RAPID_SCALE_OUT (+{target_replicas - cur_replicas} Web Pods)\n"
        f"- Risk Level: LOW (Auto-Executed by KEDA/HPA Autoscaler Policy)"
    )
    
    return {"instruction": SYSTEM_PROMPT, "input": input_text, "output": output_text, "category": "WEB_INGRESS_SURGE"}

def generate_host_security_incident():
    node_id = f"worker-node-{random.randint(1, 8):02d}"
    attacker_ip = f"198.51.100.{random.randint(10, 250)}"
    attempts = random.randint(450, 1800)
    
    input_text = (
        f"Service: Linux System & OpenSSH Daemon\n"
        f"Alert Type: AUTH_BRUTE_FORCE_DETECTED\n"
        f"Classifier Output: ANOMALY (Confidence: 0.9960)\n"
        f"Log Pattern: ['Failed password for invalid user root from {attacker_ip}', 'maximum authentication attempts exceeded', 'PAM 2 more authentication failures']\n"
        f"Host Telemetry: Node={node_id}, SSH Failed Auth Rate={attempts} attempts/min"
    )
    
    output_text = (
        f"### Chain-of-Thought (CoT) Reasoning:\n"
        f"1. Log Analysis: Continuous high-frequency SSH authentication rejections ({attempts} attempts/min) targeting administrative accounts from remote source {attacker_ip}.\n"
        f"2. Security Evaluation: Signature matches automated SSH dictionary brute-force attack attempting unauthorized host intrusion.\n"
        f"3. Containment Strategy: Enforce zero-trust ingress firewall rule on the cloud VPC gateway to drop all packets originating from the offending IP.\n\n"
        f"### Diagnosis:\n"
        f"Status: SECURITY ALERT - BRUTE FORCE ATTACK. Active automated credential-stuffing attack on management port 22 of {node_id}.\n\n"
        f"### Remediation Action:\n"
        f"- Action: block_firewall_ip(ip_address='{attacker_ip}', duration='24h', reason='Automated SSH brute-force attack on {node_id}')\n"
        f"- Cloud Scaling: SECURITY_ACL_UPDATE\n"
        f"- Risk Level: LOW (Automated Threat Neutralization)"
    )
    
    return {"instruction": SYSTEM_PROMPT, "input": input_text, "output": output_text, "category": "HOST_SECURITY_INCIDENT"}

# 3. Build Full Multi-Modal Dataset
print("\n[2/4] Synthesizing Diverse Multi-Source SRE Dataset...")

# A. HDFS Normal (1,800 samples)
sample_normal_rows = df_normal.sample(1800, random_state=42)
for _, row in sample_normal_rows.iterrows():
    dataset.append(generate_hdfs_normal(row))

# B. HDFS Network Anomaly (1,200 samples)
sample_anomaly_rows = df_anomaly.sample(1200, replace=True, random_state=42)
for _, row in sample_anomaly_rows.iterrows():
    dataset.append(generate_hdfs_network_anomaly(row))

# C. HDFS Disk Anomaly (500 samples)
sample_disk_rows = df_anomaly.sample(500, replace=True, random_state=101)
for _, row in sample_disk_rows.iterrows():
    dataset.append(generate_hdfs_disk_anomaly(row))

# D. Web Ingress Surges (800 samples)
for _ in range(800):
    dataset.append(generate_web_ingress_surge())

# E. Host Security Incidents (700 samples)
for _ in range(700):
    dataset.append(generate_host_security_incident())

# Shuffle dataset
random.shuffle(dataset)
total_samples = len(dataset)

print(f"\n[3/4] Dataset Composition ({total_samples:,} Total Instructions):")
cat_counts = pd.Series([d['category'] for d in dataset]).value_counts()
for cat, count in cat_counts.items():
    print(f"    - {cat:<25}: {count:,} samples ({count/total_samples*100:.1f}%)")

# 4. Split and Save JSON Artifacts
train_size = int(0.9 * total_samples)
train_data = dataset[:train_size]
test_data = dataset[train_size:]

output_full_path = 'nic_cloudbot_sre_dataset.json'
output_train_path = 'nic_cloudbot_sre_train.json'
output_test_path = 'nic_cloudbot_sre_test.json'

with open(output_full_path, 'w') as f:
    json.dump(dataset, f, indent=2)

with open(output_train_path, 'w') as f:
    json.dump(train_data, f, indent=2)

with open(output_test_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"\n[4/4] Saved Artifacts:")
print(f"    ✓ Complete Dataset: {output_full_path} ({len(dataset):,} samples)")
print(f"    ✓ Training Split  : {output_train_path} ({len(train_data):,} samples)")
print(f"    ✓ Validation Split: {output_test_path} ({len(test_data):,} samples)")
print("=" * 70)
print("Dataset generation complete! Ready for Kaggle/Colab SLM Fine-Tuning.")
