import numpy as np
import pandas as pd
import joblib
from classifier_tool import LogAnomalyClassifierTool
import time

print("=" * 70)
print("  NIC-CloudBot: Log Anomaly Classifier Diagnostic & Testing Suite")
print("=" * 70)

# 1. Initialize Tool
print("\n[1] Initializing LogAnomalyClassifierTool...")
tool = LogAnomalyClassifierTool()
print("    ✓ Classifier and Vectorizer loaded successfully.\n")

# 2. Test with Known Synthetic Traces
print("[2] Running Unit Tests on Synthetic Scenarios:")
print("-" * 70)

test_cases = [
    {
        "name": "Scenario A: Perfect 3-Way Normal Block Lifecycle",
        "events": ["E5", "E5", "E22", "E5", "E11", "E9", "E11", "E9", "E11", "E9", "E26", "E26", "E26", "E3", "E2", "E2", "E23", "E23", "E23", "E21", "E21", "E21"],
        "latency": 3802,
        "expected": "NORMAL"
    },
    {
        "name": "Scenario B: DataNode Connection Outage (Multiple E4 exceptions)",
        "events": ["E5", "E22", "E5", "E5", "E11", "E9", "E4", "E4", "E4", "E4", "E4", "E4", "E20"],
        "latency": 50583,
        "expected": "ANOMALY"
    },
    {
        "name": "Scenario C: Disk Write Failure (E7 writeBlock Exception + E8 Interrupted)",
        "events": ["E5", "E22", "E5", "E11", "E9", "E7", "E7", "E8", "E8", "E20"],
        "latency": 12400,
        "expected": "ANOMALY"
    },
    {
        "name": "Scenario D: Incomplete Block Replication (Missing E26 & E21)",
        "events": ["E5", "E22", "E5", "E11", "E9"],
        "latency": 1500,
        "expected": "ANOMALY"
    }
]

for tc in test_cases:
    t0 = time.perf_counter()
    res = tool.analyze_trace(tc['events'], latency=tc['latency'])
    dt_ms = (time.perf_counter() - t0) * 1000
    
    status_icon = "🟢" if res['status'] == "NORMAL" else "🔴"
    match = "✓ PASS" if res['status'] == tc['expected'] else "✗ FAIL"
    
    print(f"Test: {tc['name']}")
    print(f"  Result      : {status_icon} {res['status']} ({match})")
    print(f"  Confidence  : Anomaly Probability = {res['anomaly_probability']*100:.2f}%")
    print(f"  Suspicious  : {res['suspicious_events']}")
    print(f"  Latency     : {dt_ms:.3f} ms")
    print()

# 3. Test with Random Real Samples from Dataset
print("-" * 70)
print("[3] Testing on 5 Random Real-World Traces from Dataset:")
print("-" * 70)

df_traces = pd.read_csv('Event_traces.csv')
df_labels = pd.read_csv('anomaly_label.csv')
df = pd.merge(df_traces, df_labels, on='BlockId')

# Pick 3 random normal and 2 random anomalies
sample_normal = df[df['Label_y'] == 'Normal'].sample(3, random_state=42)
sample_anomaly = df[df['Label_y'] == 'Anomaly'].sample(2, random_state=42)
sample_df = pd.concat([sample_normal, sample_anomaly]).sample(frac=1, random_state=42)

for _, row in sample_df.iterrows():
    raw_events = [e.strip() for e in row['Features'].strip("[]").split(",") if e.strip()]
    true_label = row['Label_y'].upper()
    
    res = tool.analyze_trace(raw_events, latency=row['Latency'])
    status_icon = "🟢" if res['status'] == "NORMAL" else "🔴"
    match = "✓ MATCH" if res['status'] == true_label else "✗ MISMATCH"
    
    print(f"Block ID    : {row['BlockId']}")
    print(f"True Label  : {true_label}")
    print(f"Prediction  : {status_icon} {res['status']} ({match}) | Prob: {res['anomaly_probability']*100:.2f}%")
    print(f"Event Count : {res['total_events']} events")
    print("-" * 50)

print("\nAll tests completed!")
