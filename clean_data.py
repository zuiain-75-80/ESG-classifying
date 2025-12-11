import pandas as pd
import re

def clean_text(df):
    # Hàm loại bỏ các ký tự đặc biệt
    def remove_special_chars(text):
        return re.sub(r'[!@#$^&*]', '', text)
    
    # Hàm loại bỏ URL chứa 'http'
    def remove_links(text):
        return re.sub(r'http\S+', '', text)
    
    # Áp dụng các bước làm sạch
    df['vi_text'] = df['vi_text'].apply(lambda x: str(x).lower())  # Chuyển về chữ thường
    df['vi_text'] = df['vi_text'].apply(lambda x: remove_special_chars(x))  # Loại bỏ ký tự đặc biệt
    df['vi_text'] = df['vi_text'].apply(lambda x: remove_links(x))  # Loại bỏ URL
    
    # Loại bỏ các dòng có số từ < 8
    df = df[df['vi_text'].apply(lambda x: len(x.split()) >= 8)]
    
    return df
