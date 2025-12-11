import os
import pandas as pd
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

def interleave_by_label(df: pd.DataFrame, label_col='labels', order=[0,1,2,3]):
    grouped = {label: df[df[label_col] == label].copy().reset_index(drop=True) for label in order}
    
    max_len = max(len(grouped[label]) for label in order)
    
    interleaved = []

    for i in range(max_len):
        for label in order:
            if i < len(grouped[label]):
                interleaved.append(grouped[label].iloc[i])
    
    return pd.DataFrame(interleaved).reset_index(drop=True)
