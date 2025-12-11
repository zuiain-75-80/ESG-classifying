from config.config import ESGConfig
from models.model_trainer import ESGModelTrainer
import argparse
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description='Train ESG Classification Model')
    parser.add_argument('--train-path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--test-path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--model-name', type=str, default='vinai/phobert-base',
                       help='Model name to use')
    parser.add_argument('--resume', action='store_true',
                    help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = ESGConfig(args.train_path, args.test_path, args.resume, args.model_name)
        
        # Update config với args
        config.model.model_name = args.model_name
        config.model.num_epochs = args.epochs
        config.model.batch_size = args.batch_size
        config.model.max_length = args.max_length
        config.model.learning_rate = args.learning_rate
        
        print("Bắt đầu training ESG Classification Model")
        print(f"Data: {args.train_path}")
        print(f"Test data: {args.test_path}")
        print(f"Model: {args.model_name}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max length: {args.max_length}")
        print(f"Learning rate: {args.learning_rate}")
        
        # Training
        trainer = ESGModelTrainer(config)
        results = trainer.train_model()
        
        print(f"\nTraining hoàn thành!")
        print(f"Model đã được lưu tại: {config.paths.model_dir}")
        print(f"Results đã được lưu tại: {config.paths.results_dir}")
        print(f"Final F1 Score: {results['eval_results']['eval_f1_weighted']:.4f}")

        del trainer
        del config
        import torch
        torch.cuda.empty_cache()

        
    except (ValueError, FileNotFoundError) as e:
        print(f"Lỗi: {e}")
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        print(f"Lỗi trong quá trình training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # sys.argv = [
    #     "train.py",
    #     "--train-path", "data/environment-sentiment.csv",
    #     "--test-path", "data/environment-test.csv"
    # ]
    main()


# python train.py --train-path data/train_subset5.csv --test-path data/test.csv --model-name NlpHUST/electra-base-vn --epochs 15
# python train.py --train-path ensemble_results/combine_data/combined_data_selective_2.csv --test-path data/test.csv --model-name Fsoft-AIC/videberta-base --epochs 15
# python train.py --train-path pseudo_labeling/results/combined_data_selective_2.csv --test-path data/test.csv -- --epochs 25
# python train.py --train-path pseudo_labeling/results/combined_data_selective_3.csv --test-path data/test.csv --model-name microsoft/deberta-v3-base  --epochs 10 --resume
