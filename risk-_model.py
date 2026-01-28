# risk_model.py
import numpy as np

def measles_risk(row):
    score = 0

    score += row["under1"] * 0.35
    score += row["1_4"] * 0.30
    score += row["urban_ratio"] * 0.15
    score += (1 - row["vacc_coverage"]) * 0.15
    score += row["recent_cases"] * 0.05

    return min(score * 100, 100)
  
def predict_cases(ts, weeks=12):
    trend = np.polyfit(range(len(ts)), ts, 1)
    future = [trend[0]*i + trend[1] for i in range(len(ts), len(ts)+weeks)]
    return future
