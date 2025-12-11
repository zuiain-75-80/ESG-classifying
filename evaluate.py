from config.config import ESGConfig
from data.data_loader import ESGDataLoader
from models.esg_model import ESGModel
from utils.metrics import ESGMetrics
from utils.text_preprocessing import VietnameseTextPreprocessor
from pseudo_labeling.utils.utils import get_unique_path
import torch
import numpy as np
import argparse
import json
import os

class ESGEvaluator:
    def __init__(self, data_path: str, model_path: str = "saved_model/phobert-base"):
        self.data_path = data_path
        self.model_path = model_path
        self.config = ESGConfig("data/train.csv", data_path, resume = False, model_name = model_path) ## ở đây không dùng train_path
        self.preprocessor = VietnameseTextPreprocessor()
        
    def evaluate(self, save_results: bool = True):
        """Evaluate model trên test set"""
        
        # Load data
        data_loader = ESGDataLoader(self.config)
        df = data_loader.load_data()
        df = self.preprocessor.preprocess_dataframe(df, self.config.text_column)
        data_loader.df = df
                
        # Load model
        model = ESGModel(
            model_name=self.config.model.model_name,
            num_labels=len(self.config.labels),
        )
        model.load_model(self.model_path)

        texts = data_loader.test_df[self.config.text_column].tolist()
        true_labels = data_loader.test_df[self.config.label_column].tolist()
        
        # Predict trên test set
        print("Generating predictions...")
        batch_results = model.batch_predict(texts, batch_size=4)  # <-- truyền batch_size tại đây
        predictions = [pred["class_id"] for pred in batch_results]

        print(len(predictions))
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Compute metrics
        metrics = ESGMetrics(self.config.labels)
        eval_results = metrics.detailed_evaluation(true_labels, predictions)
        
        # Print results
        print("="*60)
        
        overall = eval_results['overall_metrics']
        print(f"accuracy: {overall['accuracy']:.4f}")
        print(f"F1 Macro: {overall['f1_macro']:.4f}")
        print(f"F1 Micro: {overall['f1_micro']:.4f}")
        print(f"F1 Weighted: {overall['f1_weighted']:.4f}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        
        print(f"\nPer-class Results:")
        per_class = eval_results['per_class_metrics']
        for label in self.config.labels:
            if label in per_class:
                metrics_data = per_class[label]
                print(f"  {label}:")
                print(f"    F1: {metrics_data['f1-score']:.4f}")
                print(f"    Precision: {metrics_data['precision']:.4f}")
                print(f"    Recall: {metrics_data['recall']:.4f}")
                print(f"    Support: {metrics_data['support']}")
        
        # Save results
        if save_results:
            results_data = {
                'model_path': self.model_path,
                'data_path': self.data_path,
                'test_size': 3000,
                'evaluation_results': eval_results['overall_metrics'],
                'per_class_results': eval_results['per_class_metrics']
            }
            
            results_path = os.path.join(self.config.paths.results_dir, f'evaluation_results.json')
            results_path = get_unique_path(results_path)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            print(f"\nEvaluation results saved to: {results_path}")
        
        return eval_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate ESG Classification Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--model-path', type=str, default="saved_model/phobert",
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    try:
        # Evaluate
        evaluator = ESGEvaluator( args.data_path, args.model_path)
        evaluator.evaluate()
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()

# python evaluate.py --data-path data/test.csv --model-path saved_model/electra-base-vn
