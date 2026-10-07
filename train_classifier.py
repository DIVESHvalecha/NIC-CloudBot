import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.sparse import hstack
import time

print("=" * 60)
print("NIC-CloudBot: Fast Log Anomaly Classifier Training")
print("=" * 60)

# 1. Load Data
print("\n[Step 1/5] Loading CSV files...")
t0 = time.time()
df_traces = pd.read_csv('Event_traces.csv')
df_labels = pd.read_csv('anomaly_label.csv')
df_templates = pd.read_csv('HDFS_templates.csv')

print(f"Loaded {len(df_traces):,} traces in {time.time() - t0:.2f}s")
print(f"Total Unique Templates: {len(df_templates)}")

# Merge with anomaly_label.csv to ensure 100% label alignment
df = pd.merge(df_traces, df_labels, on='BlockId', suffixes=('_trace', '_groundtruth'))
df['target'] = (df['Label_groundtruth'] == 'Anomaly').astype(int)

anomaly_count = df['target'].sum()
normal_count = len(df) - anomaly_count
print(f"Class Distribution: {normal_count:,} Normal ({normal_count/len(df)*100:.2f}%), {anomaly_count:,} Anomaly ({anomaly_count/len(df)*100:.2f}%)")

# 2. Extract Event Sequence Features (Bag of Events + Bigrams)
print("\n[Step 2/5] Vectorizing Event Sequences (E1 - E29)...")
t0 = time.time()
# Clean string "[E5, E22, ...]" to "E5 E22 ..."
df['clean_events'] = df['Features'].str.replace(r"[\[\],]", " ", regex=True)

# CountVectorizer capturing unigrams (E5) and bigrams (E5 E22)
vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'E\d+', lowercase=False)
X_events = vectorizer.fit_transform(df['clean_events'])
print(f"Extracted {X_events.shape[1]} event sequence features in {time.time() - t0:.2f}s")

# 3. Extract Temporal & Latency Features
print("\n[Step 3/5] Computing Temporal & Latency Statistics...")
t0 = time.time()
def parse_time_stats(interval_str):
    try:
        arr = [float(x.strip()) for x in interval_str.strip("[]").split(",") if x.strip()]
        if not arr:
            return [0.0, 0.0, 0.0, 0]
        return [float(np.mean(arr)), float(np.max(arr)), float(np.std(arr)), len(arr)]
    except:
        return [0.0, 0.0, 0.0, 0]

time_stats = np.array(df['TimeInterval'].apply(parse_time_stats).tolist())
latency = df['Latency'].fillna(0).values.reshape(-1, 1)

X_numeric = np.hstack([time_stats, latency])
X = hstack([X_events, X_numeric]).tocsr()
y = df['target'].values
print(f"Total Feature Matrix Shape: {X.shape} (Time: {time.time() - t0:.2f}s)")

# 4. Train/Test Split (Stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")

# 5. Train Classifier (HistGradientBoosting / RandomForest with balanced weighting)
print("\n[Step 4/5] Training High-Performance Classifier...")
t0 = time.time()

# Calculate class weight ratio for extreme imbalance (~97% normal vs 3% anomaly)
sample_weight = np.where(y_train == 1, normal_count / anomaly_count, 1.0)

# Random Forest with class_weight='balanced'
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Model Training Completed in {train_time:.2f}s!")

# 6. Evaluation
print("\n[Step 5/5] Evaluating on Unseen Test Data...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("                  MODEL EVALUATION REPORT                   ")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Normal (Success)', 'Anomaly (Fail)'], digits=4))
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save Model & Vectorizer Artifacts
print("\nSaving trained artifacts...")
joblib.dump(model, 'log_anomaly_classifier.pkl')
joblib.dump(vectorizer, 'event_vectorizer.pkl')

print("Artifacts saved:")
print("  - log_anomaly_classifier.pkl")
print("  - event_vectorizer.pkl")
print("=" * 60)
print("Classifier training complete and ready to be integrated as an Agent Tool!")
