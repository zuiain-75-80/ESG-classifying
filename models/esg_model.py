import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    ElectraTokenizer, ElectraForSequenceClassification,
    DebertaV2ForSequenceClassification
)
from typing import Dict, List, Any
import os
import json

MODEL_MAP = {
    # DistilBERT
    "distilbert": (DistilBertForSequenceClassification, DistilBertTokenizer),
    
    # RoBERTa & PhoBERT
    "roberta": (RobertaForSequenceClassification, AutoTokenizer),
    "phobert": (RobertaForSequenceClassification, AutoTokenizer),
    
    # XLM-R
    "xlm-roberta": (XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "visobert": (XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    
    # Electra
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
    
    # DeBERTa
    "deberta": (DebertaV2ForSequenceClassification, AutoTokenizer),

    # BERT
    "bert": (BertForSequenceClassification, BertTokenizer),
    
    
    # Fallback (default Auto)
    "auto": (AutoModelForSequenceClassification, AutoTokenizer)
}

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    for key in MODEL_MAP.keys():
        if key in name:
            return key
    return "auto"  # fallback

class ESGModel:
    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.label_names = None
    
    def initialize_model(self):
        """Khởi tạo model và tokenizer"""
        print(f"Initializing model: {self.model_name}")
        model_type = detect_model_type(self.model_name)
        ModelClass, TokenizerClass = MODEL_MAP[model_type]
        self.model = ModelClass.from_pretrained(self.model_name,
                        num_labels=self.num_labels,
                    )
        self.tokenizer = TokenizerClass.from_pretrained(self.model_name)
        print(f"Model initialized with {self.num_labels} labels")
    
    def save_model(self, save_path: str, label_names: List[str], additional_info: Dict[str, Any] = None):
        """Lưu model và metadata"""
        os.makedirs(save_path, exist_ok=True)
        
        # Lưu model và tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Lưu metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'label_names': label_names,
            'additional_info': additional_info or {}
        }
        
        with open(os.path.join(save_path, 'model_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Model đã được lưu tại: {save_path}")
    
    def load_model(self, model_path: str):
        """Load model từ đường dẫn"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path không tồn tại: {model_path}")
        print(f"Loading model from: {model_path}")

        model_type = detect_model_type(model_path)
        ModelClass, TokenizerClass = MODEL_MAP[model_type]
        self.model = ModelClass.from_pretrained(model_path,
                        num_labels=self.num_labels,
                    )
        self.tokenizer = TokenizerClass.from_pretrained(model_path)
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.label_names = metadata.get('label_names', [])
    
    def predict(self, text: str) -> Dict[str, float]:
        """Dự đoán cho một text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model chưa được khởi tạo hoặc load.")
        
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            max_length=256
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()

        results = {
                "class": self.label_names[predicted_class_idx],
                "confidence": probabilities[predicted_class_idx].item(),
                "class_id": predicted_class_idx
            }
        
        return results
    

    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model chưa được khởi tạo hoặc load.")
        
        self.model.eval()
        self.model.to(device)
        all_predictions = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)

                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                confidences, class_ids = torch.max(probs, dim=1)

                for i in range(len(batch_texts)):
                    class_id = class_ids[i].item()
                    confidence = confidences[i].item()
                    results = {
                        "class": self.label_names[class_id],
                        "confidence": confidence,
                        "class_id": class_id
                    }
                    all_predictions.append(results)


        return all_predictions

