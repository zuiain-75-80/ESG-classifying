import os
import subprocess
import time
model_names = ["vinai/phobert-base",
    "distilbert/distilbert-base-multilingual-cased",
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/roberta-base",
    "NlpHUST/electra-base-vn",
    "uitnlp/visobert",
    "Fsoft-AIC/videberta-base"
]

# train_subsets = [f"data/train_subset{i}.csv" for i in range(1, 8)]
train_file = "ensemble_results/combine_data/combined_data_selective_2.csv"
test_file = "data/test.csv"

# for idx, (model_name, train_file) in enumerate(zip(model_names, train_subsets), start=1):
#     base_model_name = model_name.split('/')[-1]
#     model_dir = f"saved_model/{base_model_name}"
#     os.makedirs(model_dir, exist_ok=True)
#     print(f"\n--- Train model {base_model_name} on {train_file} ---")
#     subprocess.run([
#         "python", "train.py",
#         "--train-path", train_file,
#         "--test-path", test_file,
#         "--model-name", model_name,
#         "--epochs", "10",
#         "--batch-size", "16",
#         "--max-length", "256",
#         "--learning-rate", "2e-5"
#     ])
#     time.sleep(5)  
# print("Training 7 models completed.")

for model_name in model_names:
    base_model_name = model_name.split('/')[-1]
    model_dir = f"saved_model/{base_model_name}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n--- Train model {base_model_name} on {train_file} ---")
    subprocess.run([
        "python", "train.py",
        "--train-path", train_file,
        "--test-path", test_file,
        "--model-name", model_name,
        "--epochs", "15",
        "--batch-size", "32",
        "--max-length", "256",
        "--learning-rate", "2e-5"
    ])
    time.sleep(5)  
print("Training 7 models completed.")
