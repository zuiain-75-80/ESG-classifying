import pandas  as pd
import numpy as np
import os, json, sys
from typing import Dict, List, Tuple, Optional, Any
from models.esg_model import ESGModel
from utils.text_preprocessing import VietnameseTextPreprocessor
from pseudo_labeling.utils.utils import get_unique_path, interleave_by_label

model_names_check = [
    "saved_model/phobert-base",
    "saved_model/vi-electra-small",
    "saved_model/visobert",
    "saved_model/videberta-base"
]

class PseudoLabler:
    def __init__(self, model_path, confidence_threshold: float = 0.8, 
                 max_samples: int = None, output_dir: str = "pseudo_labeling_output"):
        """
        Initialize Pseudo Labeler
                ...
        """
        self.model_path = model_path
        add_dir = model_path.split('/')[-1]
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.output_dir =  "pseudo_labeling/" + output_dir + "/" + add_dir
        self.preprocessor = VietnameseTextPreprocessor()

        # labels         
        self.label_names = ['Irrelevant','Evironment', 'Social','Governance']
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Labels: {self.label_names}")

    def generate_pseudo_labels(self, unlabeled_data: List[Tuple[str, Any]]) -> pd.DataFrame:
        print(f"\nGenerating pseudo labels for {len(unlabeled_data)} texts...")
        model = self.load_model(self.model_path)

        texts, id_classes = zip(*unlabeled_data)
        if self.model_path in model_names_check:
            processed_texts = [self.preprocessor.preproces_text(text) for text in texts]
        else:
            processed_texts = texts

        pseudo_labeled_data = []
        high_confidence_count = 0
        total = len(unlabeled_data)
        for batch_start in range(0, total, 4):
            batch_end = min(batch_start + 4, total)
            
            batch_texts = processed_texts[batch_start:batch_end]
            batch_original_texts = texts[batch_start:batch_end]
            batch_original_labels = id_classes[batch_start:batch_end]

            # Lọc bỏ các text rỗng (sau khi preprocess)
            valid_indices = [i for i, t in enumerate(batch_texts) if t.strip()]
            if not valid_indices:
                print(f"Batch {batch_start}-{batch_end} skipped (all empty after preprocess).")
                continue

            valid_processed_texts = [batch_texts[i] for i in valid_indices]
            predictions = model.batch_predict(valid_processed_texts)

            for idx_in_batch, prediction in zip(valid_indices, predictions):
                original_text = batch_original_texts[idx_in_batch]
                original_label = batch_original_labels[idx_in_batch]
                predicted_class_id = prediction["class_id"]
                confidence = prediction['confidence']

                if predicted_class_id == original_label and confidence >= self.confidence_threshold:
                    row = {
                        'vi_text': original_text,
                        'label_names': prediction['class'],  # Text label
                        'confidence': confidence,
                        'labels': predicted_class_id,  # Numeric ID
                        'pseudo_labeled': True,
                        'source': 'pseudo_labeling'
                    }
                    pseudo_labeled_data.append(row)
                    high_confidence_count += 1            
                # if high_confidence_count >= self.max_samples:
                #         break
            if batch_end % 800 == 0:
                print(f"Processed {batch_end}/{total} texts...")

        pseudo_df = pd.DataFrame(pseudo_labeled_data)
        pseudo_df = interleave_by_label(pseudo_df)

        if len(pseudo_df) == 0:
            print("No high-confidence pseudo labels generated. Consider lowering threshold.")
            return pseudo_df
        else:
            print(f"Processed {high_confidence_count}texts high confidence")

        # if self.max_samples and len(pseudo_df) > self.max_samples:
        #     pseudo_df = pseudo_df.iloc[:self.max_samples]
             # pseudo_df = pseudo_df.nlargest(self.max_samples, 'confidence')

        if not pseudo_df.empty:
            output_path = os.path.join(self.output_dir, f'pseudo_labels.csv')
            output_path = get_unique_path(output_path)
            pseudo_df.to_csv(output_path, index=False)
            self.save_metadata_labels(pseudo_df)
            print(f"\nPseudo data saved: {output_path}")
            print(f"Total generated: {len(pseudo_df)} samples")

            # used_pairs = set(zip(pseudo_df['vi_text'], pseudo_df['labels']))
            # remaining_data = [pair for pair in unlabeled_data if pair not in used_pairs]
            # remaining_path = os.path.join(self.output_dir, f'remain_data.csv')
            # remaining_path = get_unique_path(remaining_path)
            # remaining_df = pd.DataFrame(remaining_data, columns=["vi_text", "labels"])
            # remaining_df.to_csv(remaining_path, index = False)

        return pseudo_df
    
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path không tồn tại: {model_path}")

        model = ESGModel(
            model_name=f"{model_path}",  # Will be overridden when loading
            num_labels=len(self.label_names),
        )
        model.load_model(model_path)
        model.label_names = self.label_names
        return model
        
    def save_metadata_labels(self,pseudo_df):
        metadata = {
            'confidence_threshold': self.confidence_threshold,
            'total_pseudo_labels': len(pseudo_df),
            # 'max_samples_limit': self.max_samples,
            'labels': self.label_names,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'statistics': {
                'average_confidence': float(pseudo_df['confidence'].mean()),
                'min_confidence': float(pseudo_df['confidence'].min()),
                'max_confidence': float(pseudo_df['confidence'].max()),
                'label_distribution': pseudo_df['label_names'].value_counts().to_dict()
            }
        }
        
        metadata_path = os.path.join(self.output_dir, f'pseudo_labeling_metadata.json')
        metadata_path = get_unique_path(metadata_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved: {metadata_path}")





