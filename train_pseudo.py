import argparse
import sys
import os
import traceback

from pseudo_labeling.core.pseudo_labeler import PseudoLabler
from pseudo_labeling.core.data_combiner import DataCombiner
from pseudo_labeling.utils.utils import get_unique_path
from config.config import ESGConfig

import pandas as pd
from typing import Tuple, List, Any

def load_texts_and_labels(file_path: str) -> List[Tuple[str, Any]]:
    """
    Read csv -> Returns: List[str, Any]
    """
    try:
        df = pd.read_csv(file_path)

        if 'vi_text' not in df.columns or 'labels' not in df.columns:
            raise ValueError("File CSV phải chứa cả hai cột 'vi_text' và 'labels'.")

        df = df.dropna(subset=['vi_text'])

        df['vi_text'] = df['vi_text'].astype(str)
        return list(zip(df['vi_text'], df['labels']))

    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description=' Pseudo-Labeling')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model', default= "saved_model/phobert-base/")
    parser.add_argument('--unlabeled-file', type=str, required=True,
                       help='File containing unlabeled texts (one per line)')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                       help='Confidence threshold')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum pseudo samples to generate')
    parser.add_argument('--iterative', action='store_true',
                       help='Use iterative pseudo-labeling')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations')
    parser.add_argument('--output-dir', type=str, default='pseudo_labeling_output',
                       help='Output directory')
    parser.add_argument('--pseudo-weight', type=float, default=0.5,
                       help='Weight for pseudo samples')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze confidence distribution')
    
    args = parser.parse_args()
    
    try:                
        unlabeled_texts = load_texts_and_labels(args.unlabeled_file)
        print(f"Loaded {len(unlabeled_texts)} unlabeled texts")
        
        # Initialize pseudo labeler
        pseudo_labeler = PseudoLabler(
            model_path = args.model_path,
            confidence_threshold=args.confidence_threshold,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
        
        # Generate pseudo labels
        print("Using pseudo-labeling")
        pseudo_df = pseudo_labeler.generate_pseudo_labels(
            unlabeled_data=unlabeled_texts
        )
    
        if len(pseudo_df) == 0:
            print("No pseudo labels generated. Exiting.")
            return
                
        print(f"\nPseudo-labeling completed!")
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()  


# python train_pseudo.py  --model-path saved_model/phobert-base/environment --max-samples 1000 --unlabeled-file data/data_pseudo/raw_env.csv --confidence-threshold 0.94
# python train_pseudo.py  --model-path saved_model/deberta --max-samples 5000 --unlabeled-file pseudo_labeling/results/pseudo_labeling_output/remain_data.csv --confidence-threshold 0.94
# python train_pseudo.py  --model-path saved_model/deberta --max-samples 5000 --unlabeled-file pseudo_labeling/results/pseudo_labeling_output/remain_data_2.csv --confidence-threshold 0.94

