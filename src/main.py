import argparse
import pandas as pd
import time
import os
import shutil
import glob
from feature_extraction import DNSLogParser
from botnet_detector import BotnetDetector

from config import TRAIN_TEST_SPLIT

def get_last_exp_num(model_dir):
    existing_dirs = [os.path.basename(d) for d in glob.glob(os.path.join(model_dir, 'exp_*'))]
        
    # Extract numbers from existing directories
    max = None
    for dir_name in existing_dirs:
        cur_exp_number = int(dir_name[4:])
        if max is None or cur_exp_number > max:
            max = cur_exp_number

    return max

def feature_importance_per_family(family_results, feature_importance_by_family):
    """
    Analyze feature importance for each DGA family and format as a well-aligned table
    
    Parameters:
    family_results (dict): Results by family
    feature_importance_by_family (dict): Feature values by family
    
    Returns:
    str: Formatted table of feature importance by family
    """
    # Get all features across all families
    all_features = set()
    for family, feature_dict in feature_importance_by_family.items():
        all_features.update(feature_dict.keys())
    
    # Sort features for consistent output
    all_features = sorted(list(all_features))
    
    # Determine column widths
    family_width = max(max(len(family) for family in family_results.keys()), len("family"))
    detection_width = max(len("detection_rate"), 12)  # Minimum width for detection_rate
    feature_widths = {}
    for feature in all_features:
        # Get max width needed for this feature (header or values)
        values_width = max(
            len(f"{feature_importance_by_family[family].get(feature, 0):.4f}")
            for family in family_results.keys()
        )
        feature_widths[feature] = max(len(feature), values_width)
    
    # Create header
    header = f"{'family'.ljust(family_width)} | {'detection_rate'.ljust(detection_width)}"
    for feature in all_features:
        header += f" | {feature.ljust(feature_widths[feature])}"
    
    # Create separator line
    separator = "-" * len(header)
    
    # Create rows
    rows = []
    for family, metrics in sorted(family_results.items(), key=lambda x: x[1]['detection_rate'], reverse=True):
        detection_rate = metrics['detection_rate']
        row = f"{family.ljust(family_width)} | {f'{detection_rate:.4f}'.ljust(detection_width)}"
        
        for feature in all_features:
            value = feature_importance_by_family[family].get(feature, 0)
            # Format the value to 4 decimal places and align
            row += f" | {f'{value:.4f}'.ljust(feature_widths[feature])}"
        
        rows.append(row)
    
    # Combine everything
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    return table

def format_confusion_matrix(confusion_matrix):
    """
    Format confusion matrix as text
    
    Parameters:
    confusion_matrix (numpy.ndarray): Confusion matrix
    
    Returns:
    str: Formatted confusion matrix
    """
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    matrix_text = (
        "Confusion Matrix:\n"
        f"              | Predicted Negative | Predicted Positive\n"
        f"True Negative | {tn:18d} | {fp:18d}\n"
        f"True Positive | {fn:18d} | {tp:18d}\n"
    )
    
    return matrix_text

def save_text_results(results, family_results, feature_importance_by_family, output_dir, exp_id):
    """
    Save all results as text in the results_summary file
    
    Parameters:
    results (dict): Main evaluation results
    family_results (dict): Results by DGA family
    feature_importance_by_family (dict): Feature values by family
    output_dir (str): Directory to save results
    exp_id (str): Experiment ID
    """
    filepath = os.path.join(output_dir, f'results_summary_{exp_id}.txt')
    with open(filepath, 'w') as f:
        # General metrics
        f.write(f"Experiment ID: {exp_id}\n\n")
        f.write(f"Accuracy: {results['accuracy']:.10f}\n")
        f.write(f"Precision: {results['precision']:.10f}\n")
        f.write(f"Recall (Detection Rate): {results['recall']:.10f}\n")
        f.write(f"F1 Score: {results['f1_score']:.10f}\n")
        f.write(f"False Positive Rate: {results['false_positive_rate']:.10f}\n\n")
        
        # Add confusion matrix
        if 'confusion_matrix' in results:
            f.write(format_confusion_matrix(results['confusion_matrix']))
            f.write("\n\n")
        
        # Add family results (detection rate)
        f.write("Detection Rate by DGA Family:\n")
        for family, metrics in sorted(family_results.items(), key=lambda x: x[1]['detection_rate'], reverse=True):
            f.write(f"{family}: {metrics['detection_rate']:.10f}\n")
        
        # Add feature importance per family table
        if feature_importance_by_family:
            f.write("\n\nFeature Importance by DGA Family:\n")
            feature_table = feature_importance_per_family(family_results, feature_importance_by_family)
            f.write(feature_table)
            f.write("\n")

    print(f"\n\nResults saved to {filepath}\n\n")
    

def main():
    parser = argparse.ArgumentParser(description='IoT Botnet Detection using DNS Analysis with One-class SVM')
    parser.add_argument('--dns_log', type=str, help='Path to DNS log file')
    parser.add_argument('--dga_file', type=str, help='Path to file containing DGA domains')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model_dir', type=str, default='/app/models', help='Directory to save/load model')
    parser.add_argument('--output_dir', type=str, default='/app/results', help='Directory to save results')
    
    args = parser.parse_args()
    
    
    exp_number = get_last_exp_num(args.model_dir)

    if args.train:
        # Determine the next experiment number
        next_exp_number = 1
        if exp_number:
            next_exp_number = exp_number + 1
        exp_id = f'exp_{next_exp_number}'
    elif args.evaluate:
        exp_id = f'exp_{exp_number}'
    
    print(f"Using sequential experiment ID: {exp_id}")
    
    # Create model and output directories with experiment ID
    model_dir = os.path.join(args.model_dir, exp_id)
    output_dir = os.path.join(args.output_dir, exp_id)
    
    dns_parser = DNSLogParser()
    detector = BotnetDetector(model_dir=model_dir)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Copy config.py to the output directory to keep record of experiment parameters
    try:
        config_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
        config_dest = os.path.join(output_dir, f'config_{exp_id}.py')
        shutil.copy2(config_source, config_dest)
        print(f"Saved configuration to {config_dest}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")
    
    if not args.dns_log:
        print("No DNS log file specified. Use --dns_log to provide a log file.")
        return parser.print_help()
    
    print(f"Processing DNS logs from {args.dns_log}")
    
    # Check if file exists
    if not os.path.exists(args.dns_log):
        print(f"Error: DNS log file {args.dns_log} not found.")
        return
    
    
    dns_data = dns_parser.parse_dnsmasq_log(args.dns_log)
    
    # Sort by timestamp
    dns_data = dns_data.sort_values('timestamp')
    
    # Split data by time (80% train, 20% test)
    split_idx = int(TRAIN_TEST_SPLIT * len(dns_data))
    
    train_data = dns_data.iloc[:split_idx]
    test_benign_data = dns_data.iloc[split_idx:]
    
    train_domains = train_data['domain'].unique()
    test_benign_domains = test_benign_data['domain'].unique()
    
    print(f"Training data: {len(train_domains)} unique domains")
    print(f"Testing data (benign): {len(test_benign_domains)} unique domains")
    
    if args.train:
        # Measure training time
        start_time = time.time()
        
        # Train the model
        detector.train(train_domains)
        
        # Calculate and print elapsed time
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Save timing info to file
        with open(os.path.join(output_dir, f'timing_info_{exp_id}.txt'), 'w') as f:
            f.write(f"Experiment ID: {exp_id}\n")
            f.write(f"Training time: {train_time:.2f} seconds for {len(train_domains)} domains\n")
            f.write(f"Average time per domain: {train_time/len(train_domains):.6f} seconds\n")
            
            # Also save SVM parameters
            f.write("\nParameters used in this experiment:\n")
            try:
                from config import SVM_PARAMS, NGRAM_SIZE, NGRAM_TOP_K, FEATURE_CONFIG
                f.write(f"SVM_PARAMS: {SVM_PARAMS}\n")
                f.write(f"NGRAM_SIZE: {NGRAM_SIZE}\n")
                f.write(f"NGRAM_TOP_K: {NGRAM_TOP_K}\n")
                f.write(f"FEATURE_CONFIG: {FEATURE_CONFIG}\n")
            except ImportError:
                f.write("Could not import configuration parameters\n")
    
    if args.evaluate:
        if not args.dga_file:
            raise Exception("No DGA data provided for evaluation. Use --dga_file to provide malicious domains.")
        
        print(f"Loading DGA domains from {args.dga_file}")
        
        # Check if file exists
        if not os.path.exists(args.dga_file):
            print(f"Error: DGA file {args.dga_file} not found.")
            return
        
        # Measure evaluation time
        start_time = time.time()
        
        dga_data = pd.read_csv(args.dga_file)
        
        if 'category' not in dga_data.columns or 'family' not in dga_data.columns or 'domain' not in dga_data.columns:
            raise ValueError("Error: incompatible DGA file. Must contain 'domain', 'family', and 'category' columns.")
        
        malicious_data = dga_data[dga_data['category'] == 'dga']
        
        print(f"Loaded {len(malicious_data)} malicious domains from DGA file")
        # Group by family
        dga_families = {}
        for family, group in malicious_data.groupby('family'):
            domains = group['domain'].tolist()
            print(f"DGA family {family}: {len(domains)} domains")
            dga_families[family] = domains
        
        # Evaluate per family
        family_results, feature_importance_by_family = detector.evaluate_dga_families(dga_families)
                
        # Evaluate overall performance
        malicious_domains = malicious_data['domain'].tolist()
        results = detector.evaluate(test_benign_domains, malicious_domains)
        
        # Calculate and print elapsed time
        eval_time = time.time() - start_time
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Save timing info to file
        with open(os.path.join(output_dir, f'timing_info_{exp_id}.txt'), 'a') as f:
            f.write(f"Evaluation time: {eval_time:.2f} seconds for {len(test_benign_domains) + len(malicious_domains)} domains\n")
            f.write(f"Average evaluation time per domain: {eval_time/(len(test_benign_domains) + len(malicious_domains)):.6f} seconds\n")
                
        save_text_results(results, family_results, feature_importance_by_family, output_dir, exp_id)

if __name__ == "__main__":
    main()