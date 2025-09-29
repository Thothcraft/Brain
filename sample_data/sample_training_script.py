#!/usr/bin/env python3
"""
Sample Training Script for Thoth IoT Devices
Demonstrates federated learning with sensor data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def load_sensor_data(data_path="sample_data/sensor_data.csv"):
    """Load and preprocess sensor data"""
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} sensor readings from {data_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return None

def load_labels(labels_path="sample_data/training_labels.csv"):
    """Load training labels"""
    try:
        df = pd.read_csv(labels_path)
        print(f"Loaded {len(df)} labels from {labels_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {labels_path}")
        return None

def prepare_features(sensor_df):
    """Extract features from sensor data"""
    features = []
    
    # Basic sensor readings
    feature_cols = ['temperature', 'humidity', 'pressure', 
                   'accel_x', 'accel_y', 'accel_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z']
    
    for col in feature_cols:
        if col in sensor_df.columns:
            features.append(col)
    
    # Derived features
    if all(col in sensor_df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
        sensor_df['accel_magnitude'] = np.sqrt(
            sensor_df['accel_x']**2 + 
            sensor_df['accel_y']**2 + 
            sensor_df['accel_z']**2
        )
        features.append('accel_magnitude')
    
    if all(col in sensor_df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
        sensor_df['gyro_magnitude'] = np.sqrt(
            sensor_df['gyro_x']**2 + 
            sensor_df['gyro_y']**2 + 
            sensor_df['gyro_z']**2
        )
        features.append('gyro_magnitude')
    
    return sensor_df[features]

def train_activity_classifier(sensor_df, labels_df):
    """Train activity recognition model"""
    print("\n=== Training Activity Classifier ===")
    
    # Merge sensor data with labels
    merged_df = pd.merge(sensor_df, labels_df, on=['timestamp', 'device_id'], how='inner')
    print(f"Merged dataset size: {len(merged_df)}")
    
    # Prepare features and labels
    X = prepare_features(merged_df)
    y = merged_df['activity_label']
    
    print(f"Feature columns: {list(X.columns)}")
    print(f"Activity classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/activity_classifier.pkl")
    joblib.dump(scaler, "models/feature_scaler.pkl")
    
    print("Model saved to models/activity_classifier.pkl")
    print("Scaler saved to models/feature_scaler.pkl")
    
    return model, scaler, accuracy

def train_occupancy_detector(sensor_df, labels_df):
    """Train occupancy detection model"""
    print("\n=== Training Occupancy Detector ===")
    
    # Merge sensor data with labels
    merged_df = pd.merge(sensor_df, labels_df, on=['timestamp', 'device_id'], how='inner')
    
    # Prepare features and labels
    X = prepare_features(merged_df)
    y = merged_df['occupancy']
    
    print(f"Occupancy classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOccupancy Detection Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, "models/occupancy_detector.pkl")
    joblib.dump(scaler, "models/occupancy_scaler.pkl")
    
    print("Model saved to models/occupancy_detector.pkl")
    
    return model, scaler, accuracy

def generate_training_report(results):
    """Generate training report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "models_trained": len(results),
        "results": results,
        "data_summary": {
            "total_samples": len(results.get("sensor_data", [])),
            "devices": len(set(results.get("sensor_data", {}).get("device_id", []))),
            "features_used": results.get("features", [])
        }
    }
    
    with open("models/training_report.json", "w") as f:
        import json
        json.dump(report, f, indent=2)
    
    print(f"\nTraining report saved to models/training_report.json")
    return report

def main():
    """Main training pipeline"""
    print("=== Thoth IoT Device Training Pipeline ===")
    print(f"Started at: {datetime.now()}")
    
    # Load data
    sensor_df = load_sensor_data()
    labels_df = load_labels()
    
    if sensor_df is None or labels_df is None:
        print("Error: Could not load required data files")
        return
    
    results = {}
    
    # Train activity classifier
    try:
        activity_model, activity_scaler, activity_acc = train_activity_classifier(sensor_df, labels_df)
        results["activity_classifier"] = {
            "accuracy": activity_acc,
            "model_path": "models/activity_classifier.pkl",
            "scaler_path": "models/feature_scaler.pkl"
        }
    except Exception as e:
        print(f"Error training activity classifier: {e}")
    
    # Train occupancy detector
    try:
        occupancy_model, occupancy_scaler, occupancy_acc = train_occupancy_detector(sensor_df, labels_df)
        results["occupancy_detector"] = {
            "accuracy": occupancy_acc,
            "model_path": "models/occupancy_detector.pkl",
            "scaler_path": "models/occupancy_scaler.pkl"
        }
    except Exception as e:
        print(f"Error training occupancy detector: {e}")
    
    # Generate report
    results["sensor_data"] = sensor_df.to_dict()
    results["features"] = list(prepare_features(sensor_df).columns)
    
    report = generate_training_report(results)
    
    print(f"\n=== Training Complete ===")
    print(f"Models trained: {len([k for k in results.keys() if k.endswith('_classifier') or k.endswith('_detector')])}")
    print(f"Finished at: {datetime.now()}")

if __name__ == "__main__":
    main()
