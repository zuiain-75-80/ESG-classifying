import pandas as pd
from typing import Optional, Dict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pseudo_labeling.utils.utils import get_unique_path
import json


class DataCombiner:
    def __init__(self, output_dir: str = "ensemble_results/combine_data/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_datasets(self, original_data_path: str, pseudo_labels_path: str) -> tuple:
        """Load và copy datasets"""
        original_df = pd.read_csv(original_data_path).copy()
        pseudo_df = pd.read_csv(pseudo_labels_path).copy()
        
        print(f"   Original data: {len(original_df)} samples")
        print(f"   Pseudo data: {len(pseudo_df)} samples")
        
        return original_df, pseudo_df
    
    def add_metadata(self, original_df: pd.DataFrame, pseudo_df: pd.DataFrame, 
                     pseudo_weight: float) -> tuple:
        """Thêm metadata columns cho cả hai datasets"""
        # Metadata cho original data
        original_df['confidence'] = 1.0
        original_df['pseudo_labeled'] = False
        original_df['source'] = 'original'
        
        # Metadata cho pseudo data
        if 'confidence' not in pseudo_df.columns:
            pseudo_df['confidence'] = pseudo_weight
        if 'pseudo_labeled' not in pseudo_df.columns:
            pseudo_df['pseudo_labeled'] = True
        if 'source' not in pseudo_df.columns:
            pseudo_df['source'] = 'pseudo_labeling'
            
        return original_df, pseudo_df
    
    def _get_common_columns_and_combine(self, original_df: pd.DataFrame, 
                                       pseudo_df: pd.DataFrame) -> pd.DataFrame:
        """Tìm common columns và combine datasets"""
        original_cols = set(original_df.columns)
        pseudo_cols = set(pseudo_df.columns)
        common_cols = original_cols.intersection(pseudo_cols)
        
        print("Original columns:", original_cols)
        print("Pseudo columns:", pseudo_cols)
        print("Common columns:", common_cols)
        
        # Select only common columns
        original_subset = original_df[list(common_cols)]
        pseudo_subset = pseudo_df[list(common_cols)]
        
        # Combine datasets
        combined_df = pd.concat([original_subset, pseudo_subset], ignore_index=True)
        return combined_df
    
    def print_statistics(self, combined_df: pd.DataFrame, original_count: int, 
                         pseudo_count: int):
        """In thống kê về combined dataset"""
        print(f"Combined dataset: {len(combined_df)} samples")
        print(f"Original: {original_count} ({original_count/len(combined_df)*100:.1f}%)")
        print(f"Pseudo: {pseudo_count} ({pseudo_count/len(combined_df)*100:.1f}%)")
    
    def save_dataset(self, combined_df: pd.DataFrame, output_filename: str) -> str:
        """Lưu combined dataset"""
        combined_df.to_csv(output_filename, index=False)
        print(f"Combined data saved: {output_filename}")
        return output_filename

    def combine_datasets_with_selection(self, original_data_path: str, pseudo_labels_path: str,
                                      add_counts: Dict[int, int],
                                      label_column: str = 'labels',
                                      pseudo_weight: float = 0.5) -> pd.DataFrame:
        """Kết hợp original data với pseudo-labeled data theo số lượng được chỉ định"""
        
        original_df, pseudo_df = self.load_datasets(original_data_path, pseudo_labels_path)
        
        original_counts, pseudo_counts = self.show_label_distribution(
            original_data_path, pseudo_labels_path, label_column
        )
        #check addC-counts
        valid_selection = True
        for label, requested_count in add_counts.items():
            available = pseudo_counts.get(label, 0)
            if requested_count > available:
                print(f"Label {label}: Requested {requested_count} but only {available} available")
                valid_selection = False
            else:
                print(f"Label {label}: {requested_count} samples (available: {available})")
        if not valid_selection:
            raise ValueError("Số lượng yêu cầu vượt quá số lượng có sẵn!")
        
        selected_pseudo_dfs = []
        used_indices = set()
        total_added = 0
        for label, count in add_counts.items():
            if count > 0:
                label_data = pseudo_df[pseudo_df[label_column] == label]
                selected_data = label_data.head(count)
                selected_pseudo_dfs.append(selected_data)
                used_indices.update(selected_data.index)
                total_added += len(label_data)
                print(f"   Label {label}: Added {len(label_data)} samples")

        if selected_pseudo_dfs:
            selected_pseudo_df = pd.concat(selected_pseudo_dfs, ignore_index=True)
        else:
            selected_pseudo_df = pd.DataFrame()

        remaining_pseudo_df = pseudo_df.drop(index=used_indices)

        # Thêm metadata
        original_df, selected_pseudo_df = self.add_metadata(
            original_df, selected_pseudo_df, pseudo_weight
        )

        # Combine datasets
        if not selected_pseudo_df.empty:
            combined_df = self._get_common_columns_and_combine(original_df, selected_pseudo_df)
        else:
            combined_df = original_df

        self.print_statistics(combined_df, len(original_df), total_added)
        
        # label distribution
        label_distribution_info = {}
        final_counts = combined_df[label_column].value_counts().sort_index()
        print(f"\nFinal label distribution:")
        for label, count in final_counts.items():
            original_count = original_counts.get(label, 0)
            added_count = add_counts.get(label, 0)
            print(f"Class {label}: {count} samples = {original_count} (original) + {added_count} (pseudo)")
            label_distribution_info[str(label)] = {
                "total": int(count),
                "original": int(original_count),
                "pseudo": int(added_count)
            }
        remain_distribution = remaining_pseudo_df[label_column].value_counts().sort_index()
        print(f"\nRemaining pseudo-labeled data distribution:")
        label_distribution_info["remaining"] = {}
        for label, count in remain_distribution.items():
            print(f"Class {label}: {count} samples remaining")
            label_distribution_info["remaining"][str(label)] = int(count)

        # Save dataset
        json_filename = os.path.join(self.output_dir, f'combined_data_distribution.json')
        json_filename = get_unique_path(json_filename)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(label_distribution_info, f, indent=4, ensure_ascii=False)

        output_filename = os.path.join(self.output_dir, f'combined_data_selective.csv')
        output_filename = get_unique_path(output_filename)
        self.save_dataset(combined_df, output_filename)

        remain_output_filename = os.path.join(self.output_dir, f'remaining_pseudo_data.csv')
        remain_output_filename = get_unique_path(remain_output_filename)
        self.save_dataset(remaining_pseudo_df, remain_output_filename)

        return combined_df

    
    def analyze_combined_data(self, combined_df: pd.DataFrame, label_columns: list):
        """Phân tích combined dataset"""
        print(f"\nCombined Dataset Analysis:")
        print(f"Total samples: {len(combined_df)}")
        
        # Source distribution
        source_dist = combined_df['source'].value_counts()
        print(f"\nSource Distribution:")
        for source, count in source_dist.items():
            print(f"  {source}: {count} samples")
        
        # Weight distribution
        if 'confidence' in combined_df.columns:
            print(f"\nWeight Statistics:")
            print(f"  Average weight: {combined_df['confidence'].mean():.3f}")
            print(f"  Weight range: {combined_df['confidence'].min():.3f} - {combined_df['confidence'].max():.3f}")
        
        # Label distribution
        print(f"\nLabel Distribution:")
        for label in label_columns:
            if label in combined_df.columns:
                original_count = combined_df[combined_df['source'] == 'original'][label].sum()
                pseudo_count = combined_df[combined_df['source'] == 'pseudo_labeling'][label].sum()
                total_count = combined_df[label].sum()
                print(f"  {label}: {total_count} (Original: {original_count}, Pseudo: {pseudo_count})")
    
    def show_label_distribution(self, original_data_path: str, pseudo_labels_path: str, 
                           label_column: str = 'labels') -> tuple:
        """
        Hiển thị phân bố labels của cả 2 tập dữ liệu
        """
        original_df = pd.read_csv(original_data_path)
        pseudo_df = pd.read_csv(pseudo_labels_path)
        
        # Tính toán phân bố labels
        original_counts = original_df[label_column].value_counts().sort_index()
        pseudo_counts = pseudo_df[label_column].value_counts().sort_index()
        
        print("=" * 60)
        print("PHÂN BỐ SỐ LƯỢNG LABELS CỦA CÁC TẬP DỮ LIỆU")
        print("=" * 60)
        
        print(f"\nOriginal dataset (Total: {len(original_df)} samples):")
        for label, count in original_counts.items():
            print(f"   Class {label}: {count} samples")
        
        print(f"\nPseudo dataset (Total: {len(pseudo_df)} samples):")
        for label, count in pseudo_counts.items():
            print(f"   Class {label}: {count} samples [Available to add]")
        
        print("Bạn có thể nhập số lượng mỗi class muốn thêm từ pseudo dataset")
        print("Ví dụ: add_counts = {0: 10, 1: 5, 2: 15}")
        print("=" * 60)
        
        return original_counts.to_dict(), pseudo_counts.to_dict()
