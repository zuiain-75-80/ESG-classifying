import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from collections import Counter
from typing import List, Dict, Tuple
from pseudo_labeling.utils.utils import get_unique_path

class Ensemble:
    def __init__(self, confidence_threshold: float = 0.9, vote_threshold : int = 4, output_dir: str = "ensemble_results"):
        self.confidence_threshold = confidence_threshold
        self.vote_threshold = vote_threshold 
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_pseudo_csv_file(self, csv_paths: List[str]) -> Dict[str, List[Dict]]:
        all_predictions = {}
        for i, csv_path in enumerate(csv_paths, 1):
            model_name = csv_path.split("/")[2]  
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} không tồn tại, skip")
                continue
            df = pd.read_csv(csv_path)
            print(len(df))
            for _, row in df.iterrows():
                text = row['vi_text']
                label = row['labels']
                confidence = row['confidence']

                if text not in all_predictions:
                    all_predictions[text] = []

                all_predictions[text].append({
                    'model_name': model_name,
                    'label': label,
                    'confidence': confidence,
                    'label_name': row['label_names']
                })
        return all_predictions
    
    def vote_predictions(self, all_predictions: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Thực hiện voting cho kết quả của ensemble cho mỗi text
        """
        final_pseudo_labels = []
        for text, predictions in all_predictions.items():
            if len(predictions) < 2:
                continue
            
            #lấy vote cho mỗi label
            labels = [p["label"] for p in predictions]
            confidences = [p['confidence'] for p in predictions]

            # count vote cho từng label
            vote_counts = Counter(labels)
            most_voted_label, max_votes = vote_counts.most_common(1)[0]
            if max_votes < self.vote_threshold:
                continue
            
            # threshold avg
            same_label_confidences = [
                p['confidence'] for p in predictions 
                if p['label'] == most_voted_label
            ]
            avg_confidence = np.mean(same_label_confidences)
            if avg_confidence < self.confidence_threshold:
                continue


            label_name = next(p['label_name'] for p in predictions 
                            if p['label'] == most_voted_label)
            
            final_pseudo_labels.append({
                'vi_text': text,
                'labels': most_voted_label,
                'label_names': label_name,
                'confidence': avg_confidence,
                'vote_count': max_votes,
                'total_models': len(predictions),
                'pseudo_labeled': True,
                'source': 'ensemble_voting'
            })
        
        return final_pseudo_labels
    
    def create_ensemble_pseudo_labels(self, csv_paths: List[str], 
                                    unlabeled_file: str = None) -> pd.DataFrame:
        print(f"Loading pseudo labels from {len(csv_paths)} models...")
        all_predictions = self.load_pseudo_csv_file(csv_paths)
        print(f"Total unique texts found: {len(all_predictions)}")

        #voting for ensemble
        final_labels = self.vote_predictions(all_predictions)
        print(f"Ensemble voting completed: {len(final_labels)} pseudo labels")
        ensemble_df = pd.DataFrame(final_labels)

        if not ensemble_df.empty:
            ensemble_df = ensemble_df.sort_values('confidence', ascending=False)
            
            output_path = os.path.join(self.output_dir, 'ensemble_pseudo_labels.csv')
            output_path = get_unique_path(output_path)
            ensemble_df.to_csv(output_path, index=False)
            print(f"Ensemble pseudo labels saved: {output_path}")

        pseudo_label_info = {
            "Total pseudo labels": len(ensemble_df),
            "Average confidence": ensemble_df['confidence'].mean(),
            "Label distribution": ensemble_df['label_names'].value_counts().to_dict()
        }
        pseudo_label_info_path = os.path.join(self.output_dir, 'pseudo_label_info.json')
        pseudo_label_info_path = get_unique_path(pseudo_label_info_path)
        with open(pseudo_label_info_path, 'w') as f:
            json.dump(pseudo_label_info, f, indent=4)
        print(f"Pseudo label info saved: {pseudo_label_info_path}")

        print(f"- Total pseudo labels: {len(ensemble_df)}")
        print(f"- Average confidence: {ensemble_df['confidence'].mean():.4f}")
        print(f"- Label distribution:")
        print(ensemble_df['label_names'].value_counts())
        
        # Tạo remain data nếu có unlabeled_file
        if unlabeled_file and os.path.exists(unlabeled_file):
            self.create_remain_data(ensemble_df, unlabeled_file)
    
        return ensemble_df
    
    def create_remain_data(self, ensemble_df: pd.DataFrame, unlabeled_file: str):
        """Tạo remain data từ unlabeled data gốc"""
        original_df = pd.read_csv(unlabeled_file)
        used_texts = set(ensemble_df['vi_text'])
        
        remain_df = original_df[~original_df['vi_text'].isin(used_texts)]
        
        remain_path = os.path.join(self.output_dir, 'remain_data.csv')
        remain_path = get_unique_path(remain_path)
        remain_df.to_csv(remain_path, index=False)
        print(f"Remain data saved: {remain_path} ({len(remain_df)} samples)")

def main():
    # Đường dẫn tới 7 file CSV từ pseudo labeling
    csv_paths = [
        "pseudo_labeling/pseudo_labeling_output/bert-base-multilingual-cased/pseudo_labels.csv",
        "pseudo_labeling/pseudo_labeling_output/distilbert-base-multilingual-cased/pseudo_labels.csv", 
        "pseudo_labeling/pseudo_labeling_output/phobert-base/pseudo_labels.csv",
        "pseudo_labeling/pseudo_labeling_output/roberta-base/pseudo_labels.csv",
        "pseudo_labeling/pseudo_labeling_output/electra-base-vn/pseudo_labels.csv",
        "pseudo_labeling/pseudo_labeling_output/videberta-base/pseudo_labels.csv",
        "pseudo_labeling/pseudo_labeling_output/visobert/pseudo_labels.csv"
    ]
    
    voter = Ensemble(
        confidence_threshold=0.94,
        vote_threshold=4,  # Ít nhất 4/7 model đồng ý
        output_dir="ensemble_results"
    )
    
    ensemble_df = voter.create_ensemble_pseudo_labels(
        csv_paths=csv_paths,
        unlabeled_file=  "data/data_pseudo/overall_data.csv" #"data/data_pseudo/overall_data.csv" "ensemble_results/remain_data.csv"
    )

if __name__ == "__main__":
    main()








