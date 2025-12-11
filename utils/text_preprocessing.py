import re
import pandas as pd
from typing import Optional
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer


class VietnameseTextPreprocessor:
    def __init__(self):
        self.vi_tokenizer = ViTokenizer
    
    def preproces_text(self,text) -> str:
        """preprocess Vietnamese text"""
        if pd.isna(text):
            return ""
    
        tokens = simple_preprocess(text)
        text = ' '.join(tokens)
        text = self.vi_tokenizer.tokenize(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocess toàn bộ dataframe"""
        df = df.copy()
        df[text_column] = df[text_column].apply(self.preproces_text)
        return df