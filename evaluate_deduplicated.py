import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.sparse import hstack
import time

print("=" * 70)
print("  NIC-CloudBot: Deduplicated (Zero-Leakage) Generalization Benchmark")
print("=" * 70)

# 1. Load Data
t0 = time.time()
df_traces = pd.read_csv('Event_traces.csv')
df_labels = pd.read_csv('anomaly_label.csv')
df = pd.merge(df_traces, df_labels, on='BlockId')
df['target'] = (df['Label_y'] == 'Anomaly').astype(int)
df['clean_events'] = df['Features'].str.replace(r"[\[\],]", " ", regex=True)

total_rows = len(df)
print(f"\n[1] Total Raw Records: {total_rows:,}")

# 2. Analyze Pattern Repetition
unique_df = df.drop_duplicates(subset=['clean_events']).copy()
num_unique = len(unique_df)
repetition_rate = (1 - (num_unique / total_rows)) * 100

print(f"    Unique Sequence Patterns: {num_unique:,}")
print(f"    Pattern Repetition Rate : {repetition_rate:.2f}%")
print(f"    Unique Normal Patterns  : {sum(unique_df['target'] == 0):,}")
print(f"    Unique Anomaly Patterns : {sum(unique_df['target'] == 1):,}")

# 3. Deduplicated Split: Split purely at the UNIQUE PATTERN LEVEL
# This guarantees that the Test Set contains ZERO patterns that exist in the Train Set.
train_patterns, test_patterns = train_test_split(
    unique_df, test_size=0.25, random_state=42, stratify=unique_df['target']
)

print(f"\n[2] Deduplicated Pattern Split (75% Train / 25% Test):")
print(f"    Training on : {len(train_patterns):,} distinct structural patterns")
print(f"    Testing on  : {len(test_patterns):,} 100% UNSEEN novel patterns (Zero Leakage)")

# 4. Feature Extraction on Unique Patterns
vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'E\d+', lowercase=False)
X_train_events = vectorizer.fit_transform(train_patterns['clean_events'])
X_test_events = vectorizer.transform(test_patterns['clean_events'])

def parse_time_stats(df_subset):
    def get_stats(interval_str):
        try:
            arr = [float(x.strip()) for x in interval_str.strip("[]").split(",") if x.strip()]
            if not arr:
                return [0.0, 0.0, 0.0, 0]
            return [float(np.mean(arr)), float(np.max(arr)), float(np.std(arr)), len(arr)]
        except:
            return [0.0, 0.0, 0.0, 0]
    time_stats = np.array(df_subset['TimeInterval'].apply(get_stats).tolist())
    latency = df_subset['Latency'].fillna(0).values.reshape(-1, 1)
    return np.hstack([time_stats, latency])

X_train_num = parse_time_stats(train_patterns)
X_test_num = parse_time_stats(test_patterns)

X_train = hstack([X_train_events, X_train_num]).tocsr()
X_test = hstack([X_test_events, X_test_num]).tocsr()

y_train = train_patterns['target'].values
y_test = test_patterns['target'].values

# 5. Train Model on Unique Patterns
print("\n[3] Training Random Forest on Unique Patterns...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Evaluate on 100% Unseen Patterns
print("\n[4] Evaluating on 100% Novel Unseen Pattern Structures:")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("       ZERO-DATA-LEAKAGE EVALUATION (UNSEEN PATTERN GENERALIZATION)       ")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Normal Patterns', 'Anomaly Patterns'], digits=4))
print(f"ROC-AUC Score on Unseen Patterns : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix (Unique Pattern Level):")
print(confusion_matrix(y_test, y_pred))
print("=" * 70)

# Top generalizable rules / feature importance
feature_names = list(vectorizer.get_feature_names_out()) + ['mean_interval', 'max_interval', 'std_interval', 'step_count', 'latency']
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]

print("\n[5] Top 10 Universal Signals Learned by Classifier (Invariants):")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank}. {feature_names[idx]:<20} (Importance: {importances[idx]*100:.2f}%)")
print("=" * 70)
