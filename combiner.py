import argparse
import json
from pseudo_labeling.core.data_combiner import DataCombiner

def main():
    parser = argparse.ArgumentParser(description='Combine original data with pseudo-labeled data')
    
    parser.add_argument('--original', '-o', required=True, 
                       help='Path to original data CSV file')
    parser.add_argument('--pseudo', '-p', required=True,
                       help='Path to pseudo-labeled data CSV file')
    parser.add_argument('--output-dir', '-d', default='ensemble_results/combine_data/',
                       help='Output directory (default: ensemble_results/combine_data/)')
    
    # Mode selection
    parser.add_argument('--mode', '-m', choices=['selective', 'analysis'], default='selective',
                       help='Mode: selective (choose amounts), analysis (analyse)')
    
    # For selective mode
    parser.add_argument('--add-counts', '-a', 
                       help='JSON string specifying amounts to add per class, e.g., \'{"0": 10, "1": 20}\'')
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = DataCombiner(output_dir=args.output_dir)
    
    if args.mode == 'analysis':
        print("Showing label distribution...")
        combiner.show_label_distribution(args.original, args.pseudo, args.label_column)
        print("\nTo combine data, use:")
        print("  --mode all (combine all pseudo data)")
        print("  --mode selective --add-counts '{\"0\": 10, \"1\": 20}' (choose specific amounts)")
                
    elif args.mode == 'selective':
        if not args.add_counts:
            print("ERROR: --add-counts required for selective mode")
            print("Example: --add-counts '{\"0\": 10, \"1\": 20, \"2\": 15}'")
            return
        
        try:
            add_counts = json.loads(args.add_counts)
            add_counts = {int(k): v for k, v in add_counts.items()}# Convert string keys to int
        except json.JSONDecodeError:
            print("ERROR: Invalid JSON format for --add-counts")
            print("Example: --add-counts '{\"0\": 10, \"1\": 20, \"2\": 15}'")
            return
        
        print("Combining with selective amounts...")
        combiner.combine_datasets_with_selection(
            original_data_path=args.original,
            pseudo_labels_path=args.pseudo,
            add_counts=add_counts,
        )

if __name__ == "__main__":
    main()
# python combiner.py --original ensemble_results/combine_data/combined_data_selective.csv --pseudo ensemble_results/ensemble_pseudo_labels_2.csv --mode selective --add-counts "{\"0\": 2200, \"1\": 1000, \"2\": 584, \"3\": 580}"
