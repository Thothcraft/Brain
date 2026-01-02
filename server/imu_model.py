"""
Simple PyTorch Model for IMU Data Classification

This module defines a lightweight neural network suitable for IMU time-series data.
The model takes sequences of IMU readings (9 features: accel+gyro+mag) and outputs
class probabilities based on the number of labels in the dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np


class IMUClassifier(nn.Module):
    """
    A simple CNN+LSTM hybrid model for IMU data classification.
    
    Architecture:
    - 1D Convolutional layers to extract local patterns
    - LSTM to capture temporal dependencies
    - Fully connected layers for classification
    
    Input shape: (batch_size, sequence_length, 9)
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int,
        sequence_length: int = 100,
        input_features: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the IMU classifier.
        
        Args:
            num_classes: Number of output classes (labels)
            sequence_length: Length of input sequences
            input_features: Number of IMU features (9 for accel+gyro+mag)
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(IMUClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        # 1D Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for convolution: (batch_size, input_features, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape back for LSTM: (batch_size, sequence_length, 64)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration information.
        
        Returns:
            Dictionary containing model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "IMUClassifier",
            "architecture": "CNN+LSTM",
            "input_shape": (None, self.sequence_length, self.input_features),
            "output_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout.p if hasattr(self.dropout, 'p') else 0
        }


class IMUDataProcessor:
    """
    Utility class for processing IMU data for training.
    """
    
    @staticmethod
    def parse_imu_json(json_data: str, sequence_length: int = 100) -> np.ndarray:
        """
        Parse IMU JSON data into numpy array.
        
        Args:
            json_data: JSON string containing IMU readings
            sequence_length: Target sequence length
            
        Returns:
            Numpy array of shape (sequence_length, 9)
        """
        import json
        
        # Parse JSON lines
        lines = json_data.strip().split('\n')
        readings = []
        
        for line in lines:
            try:
                data = json.loads(line)
                imu = data['imu']
                
                # Extract features: accel (x,y,z) + gyro (x,y,z) + mag (x,y,z)
                features = [
                    imu['accel']['x'], imu['accel']['y'], imu['accel']['z'],
                    imu['gyro']['x'], imu['gyro']['y'], imu['gyro']['z'],
                    imu['mag']['x'], imu['mag']['y'], imu['mag']['z']
                ]
                readings.append(features)
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Convert to numpy array
        if not readings:
            return np.zeros((sequence_length, 9))
        
        readings = np.array(readings)
        
        # Pad or truncate to sequence_length
        if len(readings) < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - len(readings), 9))
            readings = np.vstack([readings, padding])
        elif len(readings) > sequence_length:
            # Truncate
            readings = readings[:sequence_length]
        
        return readings
    
    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize IMU data using z-score normalization.
        
        Args:
            data: Input data of shape (sequence_length, 9)
            
        Returns:
            Normalized data
        """
        # Compute mean and std for each feature
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        # Normalize
        normalized = (data - mean) / std
        
        return normalized


def create_model(num_classes: int, **kwargs) -> IMUClassifier:
    """
    Factory function to create an IMU classifier.
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        IMUClassifier instance
    """
    return IMUClassifier(num_classes=num_classes, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create a model with 3 classes
    model = create_model(num_classes=3)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy data
    batch_size = 4
    sequence_length = 100
    
    # Create random input
    dummy_input = torch.randn(batch_size, sequence_length, 9)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output logits (first sample): {output[0]}")
