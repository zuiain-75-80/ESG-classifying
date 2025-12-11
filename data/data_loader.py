import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from typing import Tuple, List
import os


class ESGDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_length: int = 256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)  # Single label

        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length',
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


class ESGDataLoader:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.train_df = None
        self.test_df = None

    def check_required_columns(self, df: pd.DataFrame):
        """Kiểm tra các columns"""
        required_columns = [self.config.text_column, self.config.label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Thiếu các cột: {missing_columns}")
        return df

    def load_data(self) -> pd.DataFrame:
        """Load và convert dữ liệu"""
        # Load data
        if (self.config.train_path and self.config.test_path and 
            os.path.exists(self.config.train_path) and os.path.exists(self.config.test_path)):
            
            self.train_df = pd.read_csv(self.config.train_path)
            self.train_df = self.check_required_columns(self.train_df)
            
            self.test_df = pd.read_csv(self.config.test_path)
            self.test_df = self.check_required_columns(self.test_df)
            
            self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)
            print(f"Loaded {len(self.train_df)} train + {len(self.test_df)} test samples")
        else:
            self.df = pd.read_csv(self.config.train_path)
            self.df = self.check_required_columns(self.df)
            self.train_df, self.test_df = self.split_data()
            
        return self.df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chia dữ liệu thành train và test"""
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Gọi load_data() trước.")
        
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.df[self.config.label_column]  # Single column stratify
        )
        
        print(f"Train: {len(self.train_df)}, Test: {len(self.test_df)}")
        return self.train_df, self.test_df
    
    def get_dataset_info(self) -> dict:
        """Thông tin về dataset"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("Chưa split dữ liệu.")
            
        info = {
            'train_path': self.config.train_path,
            'test_path': self.config.test_path,
            'train_size': len(self.train_df),
            'test_size': len(self.test_df),
            'labels': self.config.labels,  # [0, 1, 2, 3]
            'label_distribution': self.train_df[self.config.label_column].value_counts().sort_index().to_dict(),
            'total_samples': len(self.df)
        }
        
        return info
    
    def create_torch_datasets(self, tokenizer, max_length: int = 256) -> Tuple[ESGDataset, ESGDataset]:
        """Tạo PyTorch datasets"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("Chưa split dữ liệu.")
        
        train_dataset = ESGDataset(
            self.train_df[self.config.text_column],
            self.train_df[self.config.label_column],
            tokenizer,
            max_length
        )
        
        test_dataset = ESGDataset(
            self.test_df[self.config.text_column],
            self.test_df[self.config.label_column],   
            tokenizer,
            max_length
        )
        
        return train_dataset, test_dataset
