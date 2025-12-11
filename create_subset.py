import pandas as pd

def create_subsets(file_path, n_subsets=7, samples_per_subset=10000):
    df = pd.read_csv(file_path)
    
    if len(df) < samples_per_subset:
        raise ValueError(f"Dữ liệu quá ít! Cần ít nhất {samples_per_subset} mẫu để tạo mỗi subset.")

    subsets = []
    for i in range(n_subsets):
        subset = df.sample(n=samples_per_subset, random_state=42 + i).reset_index(drop=True)
        filename = f"data/train_subset{i+1}.csv"
        subset.to_csv(filename, index=False)
        subsets.append(filename)
    return subsets

if __name__ == "__main__":
    subsets = create_subsets("data/train.csv")
    print("Created subsets:", subsets)
