from transformers import TrainingArguments, Trainer
from data.data_loader import ESGDataLoader
from models.esg_model import ESGModel
from utils.metrics import ESGMetrics
from utils.text_preprocessing import VietnameseTextPreprocessor
import os
import json
import wandb
from typing import Dict, Any
from pathlib import Path
os.environ["WANDB_MODE"] = "online"  

def get_next_training_name(project_name, base_name):
    try:
        api = wandb.Api()
        runs = api.runs(project_name)
        
        # Tìm số thứ tự cao nhất
        max_index = 0
        for run in runs:
            if run.name.startswith(f"{base_name}"):
                try:
                    index = int(run.name.split("_")[1])
                    if index > max_index:
                        max_index = index
                except (ValueError, IndexError):
                    continue
        
        return f"{base_name}_{max_index + 1}"
    
    except Exception:
        return f"{base_name}_1"

model_names_check = [
    "vinai/phobert-base",
    "NlpHUST/vi-electra-small",
    "uitnlp/visobert",
    "Fsoft-AIC/videberta-base"
]

class ESGModelTrainer:
    def __init__(self, config):
        self.config = config
        self.preprocessor = VietnameseTextPreprocessor()
        
    def train_model(self, save_results: bool = True) -> Dict[str, Any]:
        print(f"Multi-Class Classification với {len(self.config.labels)} classes")

        # Load và preprocess data
        data_loader = ESGDataLoader(self.config)
        df = data_loader.load_data()
        
        # Preprocess text
        if self.config.model.model_name in model_names_check:
            df = self.preprocessor.preprocess_dataframe(df, self.config.text_column)
        data_loader.df = df
                
        # Print dataset info
        dataset_info = data_loader.get_dataset_info()
        print(f"\nDataset Info:")
        print(f"  Total samples: {dataset_info['total_samples']}")
        print(f"  Train samples: {dataset_info['train_size']}")
        print(f"  Test samples: {dataset_info['test_size']}")
        print(f"  Labels: {dataset_info['labels']}")
        print(f"  Label distribution: {dataset_info['label_distribution']}")
        
        # Khởi tạo model
        model = ESGModel(
            model_name=self.config.model.model_name,
            num_labels=len(self.config.labels),
        )

        if self.config.resume:
            model_path = Path(self.config.paths.model_dir)
            print(f"Resume=True → cố gắng load từ {model_path}")
            model.load_model(str(model_path))
            print("Load thành công!")

        else:
            model.initialize_model()
            print("Load thất bại, fallback init")

        model.label_names = self.config.label_names

        # Tạo datasets
        train_dataset, val_dataset = data_loader.create_torch_datasets(
            model.tokenizer, 
            self.config.model.max_length
        )
        
        # Metrics
        metrics = ESGMetrics(self.config.label_names)

        project_name='Training ensemble pseudo label for classify ESG'
        next_name = get_next_training_name(project_name, f"{self.config.model.model_name.split('/')[1]}")
        wandb.login(key="e7056d3a3bc855a6e0d38b8c4ff7d2ec7ce895af")
        run = wandb.init(project= project_name, 
                         name = next_name, 
                         job_type="training",
                         anonymous="allow",
                         config={
                "model": self.config.model.model_name,
                "num_labels": len(self.config.labels),
                "classification_type": "multi-class"
            }
        )

        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'{self.config.paths.model_dir}',
            num_train_epochs=self.config.model.num_epochs,
            per_device_train_batch_size=self.config.model.batch_size,
            per_device_eval_batch_size=self.config.model.batch_size,
            warmup_steps=self.config.model.warmup_steps,
            weight_decay=self.config.model.weight_decay,
            logging_dir=self.config.paths.logs_dir,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            # report_to=None,  # Disable wandb/tensorboard,
            report_to="wandb",
            run_name=f"esg_training",
            max_steps=20000
        )
        
        # Trainer
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=metrics.compute_metrics
        )
        
        # Training
        train_result = trainer.train()
        
        # Evaluation
        eval_result = trainer.evaluate()
        
        # Save model
        model.save_model(
            save_path=f'{self.config.paths.model_dir}/',
            label_names=self.config.label_names,
            additional_info={
                'data_path': self.config.train_path,
                'training_args': training_args.to_dict(),
                'train_results': train_result.metrics,
                'eval_results': eval_result,
                'dataset_info': dataset_info
            }
        )
        
        # Lưu kết quả training
        if save_results:
            self._save_training_results(train_result.metrics, eval_result, dataset_info)
        
        results = {
            'model': model,
            'trainer': trainer,
            'train_results': train_result.metrics,
            'eval_results': eval_result,
            'dataset_info': dataset_info
        }
        
        print(f"Hoàn thành training model")
        print(f"   Accuracy: {eval_result.get('eval_accuracy', 'N/A'):.4f}")
        print(f"   F1 Score: {eval_result['eval_f1_weighted']:.4f}")
        print(f"   Precision: {eval_result['eval_precision']:.4f}")
        print(f"   Recall: {eval_result['eval_recall']:.4f}")

        
        return results
    
    def _save_training_results(self, train_metrics: dict, eval_metrics: dict, dataset_info: dict):
        """Lưu kết quả training"""
        results = {
            'data_path': self.config.train_path,
            'test_path': self.config.test_path,
            'labels': self.config.label_names,
            'dataset_info': dataset_info,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'model_config': {
                'model_name': self.config.model.model_name,
                'max_length': self.config.model.max_length,
                'num_epochs': self.config.model.num_epochs,
                'batch_size': self.config.model.batch_size,
                'learning_rate': self.config.model.learning_rate,
                'warmup_steps': self.config.model.warmup_steps,
                'weight_decay': self.config.model.weight_decay
            }
        }
        if self.config.resume:
            results_path = os.path.join(self.config.paths.results_dir, f'retraining_results.json')
        else:
            results_path = os.path.join(self.config.paths.results_dir, f'training_results.json')
        
        results_path = get_unique_path(results_path) # if self.config.resume else results_path

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Kết quả training đã được lưu tại: {results_path}")

def get_unique_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    counter = 2
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1
