# risk_model.py
def predict_cases(ts, weeks=12):
    trend = np.polyfit(range(len(ts)), ts, 1)
    future = [trend[0]*i + trend[1] for i in range(len(ts), len(ts)+weeks)]
    return future
