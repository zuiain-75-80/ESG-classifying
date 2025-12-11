import os
import subprocess
import time
model_paths = [
    "saved_model/phobert-base",
    "saved_model/distilbert-base-multilingual-cased",
    "saved_model/bert-base-multilingual-cased",
    "saved_model/roberta-base",
    "saved_model/electra-base-vn",
    "saved_model/visobert",
    "saved_model/videberta-base"
]

# unlabel_file = "data/data_pseudo/overall_data.csv"
unlabel_file = "ensemble_results/remain_data.csv"
for model_path in model_paths:
    print(f"\n--- Pseudo data of model {unlabel_file}---")
    subprocess.run([
        "python", "train_pseudo.py",
        "--model-path", model_path,
        "--max-samples", "100000",
        "--unlabeled-file", unlabel_file,
        "--confidence-threshold", "0.91",
    ])
    time.sleep(5)  
print("Pseudo data 7 models completed.")