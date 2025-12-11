# ESG Text Classification with Pseudo-Labeling

This project performs ESG (Environmental, Social, Governance) text classification using Transformer-based models combined with Pseudo-labeling and Ensemble Learning techniques.
## 🎯 Overview

The goal of this project is to classify Vietnamese text into ESG-related categories with 4 classes:
- **Class 0**: Irrelevant (Not relevant OR Neutral)
- **Class 1**: Environment (Môi trường) 
- **Class 2**: Social (Xã hội)
- **Class 3**: Governance (Quản trị)

### Đặc điểm chính:
- **Multi-model training**: Supports 7 different Transformer models
- **Pseudo-labeling with Ensemble Learning**: Automatically labels unlabeled data using ensemble learning techniques

## 📁 Cấu trúc dự án
```bash

├── configg/
│   └── configg.py                      
│
├── data/
│   ├── train.csv                       # Main training data
│   ├── test.csv                        # Test data
│   ├── train_subset1.csv               # Training data subsets
│   ├── ...                             # Additional subsets (train_subset2.csv - train_subset7.csv)
│   └── data_pseudo/
│       └── overall_data.csv           # Unlabeled data
│
├── models/
│   ├── esg_model.py                   
│   └── model_trainer.py               # Model training
│
├── pseudo_labeling/
│   ├── core/
│   │   ├── pseudo_labeler.py          # Logic pseudo-labeling
│   │   ├── data_combiner.py           # Combine original and pseudo-labeled data
│   │   └── ensemble.py                # Ensemble
│   └── utils/
│       └── utils.py  
│  └── pseudo_labeling_output/         # Pseudo-labeling outputs from each model
│
├── saved_model/                       # save model
│   ├── phobert-base/
│   ├── bert-base-multilingual-cased/
│   ├── distilbert-base-multilingual-cased/
│   ├── roberta-base/
│   ├── electra-base-vn/
│   ├── visobert/
│   └── videberta-base/
│
├── utils/
│   ├── metrics.py                     # Evaluation metric functions
│   └── text_preprocessing.py          
│
├── results/                           # Training results
│   ├── phobert-base/
│   ├── bert-base-multilingual-cased/
│   ├── distilbert-base-multilingual-cased/
│   ├── roberta-base/
│   ├── electra-base-vn/
│   ├── visobert/
│   └── videberta-base/
│
├── train.py                           
├── train_pseudo.py                    # Pseudo-labeling for a single model
├── train_multiple_models.py           # Train multiple models
├── pseudo_multiple_models.py          # Pseudo-labeling with multiple models
├── evaluate.py                        # Evaluate on test set
├── visualization.ipynb                # Visualize data and results
├── create_subset.py                   # Generate data subsets
├── combiner.py                        # Combine original and pseudo-labeled data
└── clean_data.py                      # Clean and preprocess data
```

## 🚀 How to Run

### 1. Train a Single Model

```bash
python train.py
--train-path data/train.csv
--test-path data/test.csv
--model-name vinai/phobert-base
--epochs 10
--batch-size 16
--max-length 256
--learning-rate 2e-5
```

### 2. Train Multiple Models Simultaneously
```bash
python train_multiple_models.py
``` 
** lưu ý: thay đổi tuỳ chọn trong code


### 3. Pseudo-labeling with a Single Model
```bash
python train_pseudo.py
--model-path saved_model/phobert-base
--unlabeled-file data/unlabeled.csv
--configdence-threshold 0.9
--max-samples 1000
```

### 4. Pseudo-labeling with Multiple Models
```bash
python pseudo_multiple_models.py
```


### 5. Ensemble voting
```bash
python pseudo_labeling/core/ensemble.py
```

## 📊 Quy trình Pseudo-labeling

1. **Generate Pseudo Labels**: Trained models predict on unlabeled data
2. **Filter by Confidence**: Retain only predictions with confidence > threshold
3. **Ensemble Voting**: Combine predictions from multiple models
4. **Data Combination**: Merge pseudo-labeled data with original data
5. **Iterative Training**: Re-train models using the combined dataset

## 📈 Metrics và Evaluation

This project uses the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## 🔍 Monitoring và Logging

- **Hyperparameters and metrics**: Tracking config và metrics
- **JSON Metadata**: LStore detailed logs of training and pseudo-labeling
- **CSV Results**: export data

**Before running the project, set up the virtual environment:
```bash
conda create -n bert-test python=3.10
conda activate bert-test
pip install -r requirements.txt
```