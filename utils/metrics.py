import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix, accuracy_score
)
from typing import Dict, Any

class ESGMetrics:
    def __init__(self, label_names: list):
        self.label_names = label_names
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics cho multi-class classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_micro': f1_score(labels, predictions, average='micro'), 
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'accuracy': accuracy_score(labels, predictions)
        }
    
    def detailed_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Đánh giá chi tiết cho multi-class"""
        if y_pred.ndim > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        # Overall metrics
        metrics = {
            'f1_macro': f1_score(y_true, y_pred_classes, average='macro'),
            'f1_micro': f1_score(y_true, y_pred_classes, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred_classes, average='weighted'),
            'precision': precision_score(y_true, y_pred_classes, average='weighted'),
            'recall': recall_score(y_true, y_pred_classes, average='weighted'),
            'accuracy': accuracy_score(y_true, y_pred_classes)
        }
        
        # Per-class metrics
        report = classification_report(
            y_true, y_pred_classes, 
            target_names=self.label_names,
            output_dict=True
        )
        
        return {
            'overall_metrics': metrics,
            'per_class_metrics': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred_classes)
        }
