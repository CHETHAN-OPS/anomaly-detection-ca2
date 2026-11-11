import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and train model on startup
DATA_PATH = 'dataset.csv'

def get_anomaly_results():
    if not os.path.exists(DATA_PATH):
        # Generate synthetic fallback data if dataset is missing
        np.random.seed(42)
        data = pd.DataFrame({
            'Transaction_ID': range(1, 101),
            'Amount': np.random.exponential(scale=50, size=100),
            'Frequency': np.random.poisson(lam=3, size=100)
        })
        # Inject artificial anomalies
        data.loc[10, 'Amount'] = 1200.0
        data.loc[45, 'Amount'] = 1500.0
    else:
        data = pd.read_csv(DATA_PATH)

    # Select numerical features for unsupervised detection
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Transaction_ID' in numeric_cols:
        numeric_cols.remove('Transaction_ID')
        
    X = data[numeric_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Unsupervised Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X_scaled) # Returns 1 for normal, -1 for anomaly
    
    data['Anomaly_Score'] = model.decision_function(X_scaled)
    data['Status'] = np.where(preds == -1, 'Anomaly', 'Normal')
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def fetch_data():
    df = get_anomaly_results()
    anomalies_only = df[df['Status'] == 'Anomaly']
    
    response = {
        "total_records": len(df),
        "total_anomalies": len(anomalies_only),
        "records": df.to_dict(orient='records')
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
