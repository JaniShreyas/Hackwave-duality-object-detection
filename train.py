import argparse
import os
import sys
import datetime
import json
import itertools
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Default hyperparameter ranges for tuning
HP_RANGES = {
    # Original hyperparameters
    'epochs': [10],
    'mosaic': [0.0, 0.1, 0.3, 0.5],
    'optimizer': ['AdamW', 'SGD', 'Adam'],
    'momentum': [0.2, 0.5, 0.9],
    'lr0': [0.001, 0.01, 0.0001],
    'lrf': [0.0001, 0.001, 0.00001],
    'single_cls': [False],
    
    # Basic augmentation control
    'augment': [True],  # Enable augmentations
    
    # Geometric transformations
    'fliplr': [0.0, 0.5],  # Horizontal flip probability
    'flipud': [0.0],  # Vertical flip probability (typically 0 for natural images)
    'degrees': [0.0, 10.0, 20.0],  # Rotation (+/- deg)
    'translate': [0.0, 0.1, 0.2],  # Translation (+/- fraction)
    'scale': [0.0, 0.3, 0.5],  # Scale (+/- gain)
    'shear': [0.0, 5.0],  # Shear (+/- deg)
    'perspective': [0.0, 0.0001],  # Perspective distortion (+/- fraction)
    
    # Color transformations
    'hsv_h': [0.0, 0.015, 0.03],  # HSV-Hue augmentation (fraction)
    'hsv_s': [0.0, 0.4, 0.7],  # HSV-Saturation augmentation (fraction)
    'hsv_v': [0.0, 0.2, 0.4],  # HSV-Value augmentation (fraction)
    
    # Advanced augmentations
    'mixup': [0.0, 0.1],  # Mixup probability
    'copy_paste': [0.0]  # Segment copy-paste probability
}

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def run_training(hyperparams, this_dir, run_id):
    """Run a single training with given hyperparameters."""
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=hyperparams['epochs'],
        device=0,
        single_cls=hyperparams['single_cls'],
        mosaic=hyperparams['mosaic'],
        # Added augmentations
        augment=hyperparams.get('augment', True),       # Random augmentations
        fliplr=hyperparams.get('fliplr', 0.5),          # Horizontal flip probability
        flipud=hyperparams.get('flipud', 0.0),          # Vertical flip probability
        degrees=hyperparams.get('degrees', 0.0),        # Rotation (+/- deg)
        translate=hyperparams.get('translate', 0.1),    # Translation (+/- fraction)
        scale=hyperparams.get('scale', 0.5),            # Scale (+/- gain)
        shear=hyperparams.get('shear', 0.0),            # Shear (+/- deg)
        perspective=hyperparams.get('perspective', 0.0), # Perspective distortion (+/- fraction)
        hsv_h=hyperparams.get('hsv_h', 0.015),          # HSV-Hue augmentation (fraction)
        hsv_s=hyperparams.get('hsv_s', 0.7),            # HSV-Saturation augmentation (fraction)
        hsv_v=hyperparams.get('hsv_v', 0.4),            # HSV-Value augmentation (fraction)
        mixup=hyperparams.get('mixup', 0.0),            # Mixup probability
        copy_paste=hyperparams.get('copy_paste', 0.0),  # Segment copy-paste probability
        # Original parameters
        optimizer=hyperparams['optimizer'],
        lr0=hyperparams['lr0'],
        lrf=hyperparams['lrf'],
        momentum=hyperparams['momentum'],
        project=os.path.join(this_dir, "results"),
        name=f"tuning_run_{run_id}"
    )
    
    # Get mAP50 from training results
    if hasattr(results, 'results_dict') and 'metrics/mAP50(B)' in results.results_dict:
        map50 = results.results_dict['metrics/mAP50(B)']
    else:
        # Try to get mAP50 from the last epoch's results
        try:
            map50 = round(results.metrics.get('map50', 0), 4)
        except (AttributeError, KeyError):
            map50 = "unknown"
    
    if isinstance(map50, str):
        try:
            map50 = float(map50)
        except ValueError:
            map50 = 0.0
    
    # Create a results dictionary
    result_data = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mAP50": map50,
        "hyperparameters": hyperparams,
        "metrics": results.metrics if hasattr(results, 'metrics') else {}
    }
    
    return result_data

def grid_search(param_ranges, max_combinations=None):
    """Generate all combinations of hyperparameters or a random subset."""
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    all_combinations = list(itertools.product(*values))
    total_combinations = len(all_combinations)
    
    if max_combinations and max_combinations < total_combinations:
        print(f"Total possible combinations: {total_combinations}, selecting {max_combinations} random samples")
        selected_combinations = random.sample(all_combinations, max_combinations)
    else:
        print(f"Total combinations: {total_combinations}")
        selected_combinations = all_combinations
    
    # Convert combinations to dictionaries
    param_dicts = []
    for combo in selected_combinations:
        param_dict = {keys[i]: combo[i] for i in range(len(keys))}
        param_dicts.append(param_dict)
    
    return param_dicts

def plot_results(results_df, output_dir):
    """Create visualizations of hyperparameter tuning results."""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    create_directory_if_not_exists(plots_dir)
    
    # Plot mAP50 for each run
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(results_df)), results_df['mAP50'], color='skyblue')
    plt.xlabel('Run ID')
    plt.ylabel('mAP50')
    plt.title('mAP50 Across Hyperparameter Tuning Runs')
    plt.xticks(range(len(results_df)), results_df['run_id'], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'map50_by_run.png'))
    
    # Plot impact of key hyperparameters on mAP50
    numeric_params = ['epochs', 'mosaic', 'momentum', 'lr0', 'lrf']
    
    # Create subplots for each numeric parameter
    fig, axes = plt.subplots(len(numeric_params), 1, figsize=(10, 4*len(numeric_params)))
    
    for i, param in enumerate(numeric_params):
        if param in results_df.columns:
            axes[i].scatter(results_df[param], results_df['mAP50'])
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('mAP50')
            axes[i].set_title(f'Impact of {param} on mAP50')
            
            # Try to fit a trend line if we have numerical data
            try:
                z = np.polyfit(results_df[param], results_df['mAP50'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(results_df[param]), max(results_df[param]), 100)
                axes[i].plot(x_range, p(x_range), "r--")
            except Exception as e:
                print(f"Could not fit trend line for {param}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hyperparameter_impact.png'))
    
    # For categorical parameters (like optimizer), create bar plots
    for param in results_df.columns:
        if param not in numeric_params and param != 'run_id' and param != 'mAP50':
            plt.figure(figsize=(10, 6))
            grouped = results_df.groupby(param)['mAP50'].mean().reset_index()
            plt.bar(grouped[param], grouped['mAP50'], color='skyblue')
            plt.xlabel(param)
            plt.ylabel('Average mAP50')
            plt.title(f'Impact of {param} on mAP50')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{param}_impact.png'))
    
    plt.close('all')

def save_best_model(results_df, this_dir, output_dir):
    """Save the best model based on mAP50."""
    best_run = results_df.loc[results_df['mAP50'].idxmax()]
    best_run_id = best_run['run_id']
    best_map50 = best_run['mAP50']
    
    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base filename
    base_filename = f"{timestamp}_mAP50_{best_map50:.4f}_best_from_tuning"
    
    # Path to best model from the tuning run
    best_model_path = os.path.join(this_dir, "results", f"tuning_run_{best_run_id}", "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        import shutil
        weights_filename = os.path.join(output_dir, "weights", f"{base_filename}_best.pt")
        shutil.copy(best_model_path, weights_filename)
        print(f"Best model weights saved to: {weights_filename}")
        
        # Save best hyperparameters to a separate file
        best_hp_file = os.path.join(output_dir, "logs", f"{base_filename}_best_hyperparams.json")
        with open(best_hp_file, 'w') as f:
            json.dump({
                "best_run_id": best_run_id,
                "mAP50": float(best_map50),
                "hyperparameters": {
                    key: (value if key != 'single_cls' else bool(value))
                    for key, value in best_run.items() 
                    if key in HP_RANGES.keys()
                }
            }, f, indent=4)
        print(f"Best hyperparameters saved to: {best_hp_file}")
    else:
        print(f"Warning: Could not find best model for run {best_run_id} at {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for YOLOv8")
    
    # Tuning method and limits
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random',
                        help='Tuning method: grid search or random search')
    parser.add_argument('--max_runs', type=int, default=10,
                        help='Maximum number of tuning runs to perform')
    
    # Hyperparameter ranges (optional, can override defaults)
    for param, values in HP_RANGES.items():
        if isinstance(values[0], bool):
            parser.add_argument(f'--{param}', nargs='+', type=lambda x: x.lower() == 'true', 
                               default=values, help=f'Values for {param}')
        elif isinstance(values[0], float):
            parser.add_argument(f'--{param}', nargs='+', type=float, default=values, 
                               help=f'Values for {param}')
        elif isinstance(values[0], int):
            parser.add_argument(f'--{param}', nargs='+', type=int, default=values, 
                               help=f'Values for {param}')
        else:
            parser.add_argument(f'--{param}', nargs='+', type=str, default=values, 
                               help=f'Values for {param}')
    
    args = parser.parse_args()
    
    # Get current directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    
    # Create directories for results
    output_dir = os.path.join(this_dir, "results")
    logs_dir = os.path.join(output_dir, "logs")
    weights_dir = os.path.join(output_dir, "weights")
    create_directory_if_not_exists(logs_dir)
    create_directory_if_not_exists(weights_dir)
    
    # Create parameter ranges dictionary from args
    param_ranges = {}
    for param in HP_RANGES.keys():
        param_values = getattr(args, param, None)
        if param_values is not None:
            param_ranges[param] = param_values
    
    # Generate parameter combinations
    if args.method == 'grid':
        param_combinations = grid_search(param_ranges)
    else:  # random search
        param_combinations = grid_search(param_ranges, args.max_runs)
    
    # Limit the number of runs if needed
    if len(param_combinations) > args.max_runs:
        param_combinations = param_combinations[:args.max_runs]
    
    # Run hyperparameter tuning
    all_results = []
    
    print(f"Starting hyperparameter tuning with {len(param_combinations)} runs...")
    for i, params in enumerate(param_combinations):
        print(f"\nRun {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        # Run training with these parameters
        result = run_training(params, this_dir, i+1)
        all_results.append(result)
        
        # Save intermediate results
        intermediate_file = os.path.join(logs_dir, f"tuning_results_intermediate.json")
        with open(intermediate_file, 'w') as f:
            json.dump(all_results, f, indent=4)
    
    # Process and analyze results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(logs_dir, f"tuning_results_{timestamp}.json")
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"All tuning results saved to: {final_results_file}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame([{
        'run_id': r['run_id'],
        'mAP50': float(r['mAP50']) if isinstance(r['mAP50'], (int, float)) or (isinstance(r['mAP50'], str) and r['mAP50'].replace('.', '', 1).isdigit()) else 0.0,
        **r['hyperparameters']
    } for r in all_results])
    
    # Generate plots
    if len(results_df) > 0:
        plot_results(results_df, output_dir)
        
        # Save the best model
        save_best_model(results_df, this_dir, output_dir)
        
        # Print results summary
        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Total runs: {len(results_df)}")
        if len(results_df) > 0:
            best_idx = results_df['mAP50'].idxmax()
            print(f"Best mAP50: {results_df.loc[best_idx, 'mAP50']:.4f} (Run {results_df.loc[best_idx, 'run_id']})")
            print("Best hyperparameters:")
            for param in HP_RANGES.keys():
                if param in results_df.columns:
                    print(f"  {param}: {results_df.loc[best_idx, param]}")
    else:
        print("No valid results found.")