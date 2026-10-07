import joblib
import numpy as np
from scipy.sparse import hstack

class LogAnomalyClassifierTool:
    """
    Inference Tool for NIC-CloudBot SLM Multi-Agent System.
    Loads trained RandomForest/HistGBM classifier and vectorizer.
    """
    def __init__(self, model_path='log_anomaly_classifier.pkl', vectorizer_path='event_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def analyze_trace(self, event_sequence: list, time_intervals: list = None, latency: float = 0.0) -> dict:
        """
        Takes an event list (e.g. ['E5', 'E22', 'E4', 'E4']) and returns diagnostic health report.
        """
        clean_event_str = " ".join(event_sequence)
        x_events = self.vectorizer.transform([clean_event_str])
        
        # Calculate temporal stats
        if time_intervals and len(time_intervals) > 0:
            arr = np.array(time_intervals, dtype=float)
            mean_int, max_int, std_int, count_int = float(np.mean(arr)), float(np.max(arr)), float(np.std(arr)), len(arr)
        else:
            mean_int, max_int, std_int, count_int = 0.0, 0.0, 0.0, len(event_sequence)
            
        x_numeric = np.array([[mean_int, max_int, std_int, count_int, float(latency)]])
        x_input = hstack([x_events, x_numeric]).tocsr()
        
        prob = float(self.model.predict_proba(x_input)[0, 1])
        prediction = bool(prob > 0.5)
        
        suspicious_events = [e for e in set(event_sequence) if e in ['E4', 'E7', 'E8', 'E20']]
        
        return {
            "status": "ANOMALY" if prediction else "NORMAL",
            "anomaly_probability": round(prob, 4),
            "total_events": len(event_sequence),
            "suspicious_events": suspicious_events,
            "requires_investigation": prediction or (prob > 0.35)
        }

if __name__ == "__main__":
    tool = LogAnomalyClassifierTool()
    
    # Test Normal trace
    normal_trace = ["E5", "E22", "E5", "E11", "E9", "E26", "E2", "E23", "E21"]
    print("Normal Test Trace:", tool.analyze_trace(normal_trace, latency=3802))
    
    # Test Anomaly trace (with multiple E4 exceptions)
    anomaly_trace = ["E5", "E22", "E5", "E11", "E9", "E4", "E4", "E4", "E4", "E20"]
    print("Anomaly Test Trace:", tool.analyze_trace(anomaly_trace, latency=50458))
