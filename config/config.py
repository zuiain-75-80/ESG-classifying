import os
from dataclasses import dataclass
from typing import Dict, List, Optional


class ModelConfig:
    def __init__(self, model_name="vinai/phobert-base", max_length=256, num_epochs=10,
                 batch_size=16, learning_rate=2e-5, warmup_steps=500, weight_decay=0.01):
        self.model_name = model_name
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
    @property
    def short_name(self):
        return self.model_name.split("/")[-1]
    

@dataclass
class PathConfig:
    model_dir: str = "saved_model/phobert"
    logs_dir: str = "logs"
    results_dir: str = "results/phobert"
    
    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


class ESGConfig:
    def __init__(self, train_path: str, test_path: str, resume, model_name : str = "vinai/phobert-base"):
        self.train_path = train_path
        self.test_path = test_path
        self.model = ModelConfig(model_name=model_name)
        short_name = self.model.short_name
        self.paths = PathConfig(
            model_dir=f"saved_model/{short_name}",
            logs_dir="logs",
            results_dir=f"results/{short_name}")
        self.test_size = 0.2
        self.random_state = 42
        self.text_column = "vi_text"
        self.label_column = "labels"
        self.resume = resume

        # Text labels mapping (Input format)
        self.label_mappings = {
            'Irrelevant': 0,
            'Evironment': 1, 
            'Social': 2,
            'Governance': 3,
        }
        self.label_names = list(self.label_mappings.keys())
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"File not found: {train_path}")
        
        self.labels = [0, 1, 2, 3]
        
        print(f"Config for {len(self.labels)} labels")