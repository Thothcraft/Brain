"""
IMU Model Training Script

This module handles training of the IMU classifier model for cloud training.
It integrates with the existing dataset infrastructure and provides a simple
training loop with metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import asyncio
from sqlalchemy.orm import Session

from imu_model import IMUClassifier, IMUDataProcessor, create_model
from db import TrainingJob, TrainedModel, DatasetFile, File
import logging

logger = logging.getLogger(__name__)


class IMUTrainer:
    """
    Handles training of IMU models with dataset files.
    """
    
    def __init__(
        self,
        model: IMUClassifier,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Initialize the trainer.
        
        Args:
            model: IMU classifier model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def prepare_data(
        self,
        dataset_files: List[DatasetFile],
        files_content: Dict[int, bytes],
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """
        Prepare training and validation data from dataset files.
        
        Args:
            dataset_files: List of dataset file entries with labels
            files_content: Dictionary mapping file_id to file content
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader, label_mapping)
        """
        # Group files by label
        label_groups = {}
        label_mapping = {}
        label_idx = 0
        
        for df in dataset_files:
            label = df.label
            if label not in label_groups:
                label_groups[label] = []
                label_mapping[label] = label_idx
                label_idx += 1
            label_groups[label].append(df.file_id)
        
        # Process each file
        all_sequences = []
        all_labels = []
        
        for label, file_ids in label_groups.items():
            label_idx = label_mapping[label]
            
            for file_id in file_ids:
                if file_id not in files_content:
                    logger.warning(f"File {file_id} not found in content")
                    continue
                
                # Parse IMU data
                content = files_content[file_id].decode('utf-8')
                sequence = IMUDataProcessor.parse_imu_json(content)
                
                if sequence is not None and len(sequence) > 0:
                    # Normalize the data
                    sequence = IMUDataProcessor.normalize_data(sequence)
                    all_sequences.append(sequence)
                    all_labels.append(label_idx)
        
        if not all_sequences:
            raise ValueError("No valid IMU sequences found in dataset")
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(all_sequences))
        y = torch.LongTensor(np.array(all_labels))
        
        # Split into train and validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, label_mapping
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            job_id: Training job ID for tracking
            
        Returns:
            Training results dictionary
        """
        best_val_accuracy = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log metrics
            logger.info(
                f"Job {job_id} - Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Track best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
        
        results = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch
        }
        
        return results


async def run_cloud_training(
    job_id: str,
    dataset_id: int,
    model_type: str,
    config: Dict[str, Any],
    db: Session
) -> Dict[str, Any]:
    """
    Run cloud training for IMU model.
    
    Args:
        job_id: Training job ID
        dataset_id: Dataset ID
        model_type: Type of model to train
        config: Training configuration
        db: Database session
        
    Returns:
        Training results
    """
    try:
        # Get dataset files
        dataset_files = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id
        ).all()
        
        if not dataset_files:
            raise ValueError("No files found in dataset")
        
        # Get file contents
        files_content = {}
        for df in dataset_files:
            file_record = db.query(File).filter(File.fileId == df.file_id).first()
            if file_record and file_record.content:
                files_content[df.file_id] = file_record.content
        
        # Get unique labels
        unique_labels = list(set(df.label for df in dataset_files))
        num_classes = len(unique_labels)
        
        if num_classes < 2:
            raise ValueError("Need at least 2 different labels for classification")
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(
            num_classes=num_classes,
            sequence_length=config.get('sequence_length', 100),
            hidden_dim=config.get('hidden_dim', 64),
            dropout=config.get('dropout', 0.3)
        )
        
        # Create trainer
        trainer = IMUTrainer(
            model=model,
            device=device,
            learning_rate=config.get('learning_rate', 0.001),
            batch_size=config.get('batch_size', 32)
        )
        
        # Prepare data
        train_loader, val_loader, label_mapping = trainer.prepare_data(
            dataset_files=dataset_files,
            files_content=files_content,
            validation_split=config.get('validation_split', 0.2)
        )
        
        # Train model
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get('epochs', 10),
            job_id=job_id
        )
        
        # Save model
        model_state = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_model_info(),
            'label_mapping': label_mapping,
            'training_results': results
        }
        
        # Convert to bytes for storage
        import pickle
        model_bytes = pickle.dumps(model_state)
        
        # Save to database
        trained_model = TrainedModel(
            user_id=1,  # TODO: Get from job
            job_id=job_id,
            name=f"IMU_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            architecture=model_type,
            accuracy=int(results['best_val_accuracy'] * 100),  # Store as percentage * 100
            size_bytes=len(model_bytes),
            model_data=model_bytes,
            config=json.dumps(config)
        )
        
        db.add(trained_model)
        db.commit()
        
        logger.info(f"Training completed for job {job_id}. Best accuracy: {results['best_val_accuracy']:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}")
        raise


# Test the training function
if __name__ == "__main__":
    # Create dummy data for testing
    print("Testing IMU model training...")
    
    # Create model
    model = create_model(num_classes=3)
    trainer = IMUTrainer(model)
    
    # Create dummy data
    batch_size = 32
    sequence_length = 100
    n_samples = 100
    
    X = torch.randn(n_samples, sequence_length, 9)
    y = torch.randint(0, 3, (n_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Test training
    loss, acc = trainer.train_epoch(loader)
    print(f"Test - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
