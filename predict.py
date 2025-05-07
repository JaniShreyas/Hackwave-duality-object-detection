# from ultralytics import YOLO
# from pathlib import Path
# import cv2
# import os
# import yaml


# # Function to predict and save images
# def predict_and_save(model, image_path, output_path, output_path_txt):
#     # Perform prediction
#     results = model.predict(image_path,conf=0.5)

#     result = results[0]
#     # Draw boxes on the image
#     img = result.plot()  # Plots the predictions directly on the image

#     # Save the result
#     cv2.imwrite(str(output_path), img)
#     # Save the bounding box data
#     with open(output_path_txt, 'w') as f:
#         for box in result.boxes:
#             # Extract the class id and bounding box coordinates
#             cls_id = int(box.cls)
#             x_center, y_center, width, height = box.xywh[0].tolist()
            
#             # Write bbox information in the format [class_id, x_center, y_center, width, height]
#             f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


# if __name__ == '__main__': 

#     this_dir = Path(__file__).parent
#     os.chdir(this_dir)
#     with open(this_dir / 'yolo_params.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#         if 'test' in data and data['test'] is not None:
#             images_dir = Path(data['test']) / 'images'
#         else:
#             print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
#             exit()
    
#     # check that the images directory exists
#     if not images_dir.exists():
#         print(f"Images directory {images_dir} does not exist")
#         exit()

#     if not images_dir.is_dir():
#         print(f"Images directory {images_dir} is not a directory")
#         exit()
    
#     if not any(images_dir.iterdir()):
#         print(f"Images directory {images_dir} is empty")
#         exit()

#     # Load the YOLO model
#     detect_path = this_dir / "runs" / "detect"
#     train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
#     if len(train_folders) == 0:
#         raise ValueError("No training folders found")
#     idx = 0
#     if len(train_folders) > 1:
#         choice = -1
#         choices = list(range(len(train_folders)))
#         while choice not in choices:
#             print("Select the training folder:")
#             for i, folder in enumerate(train_folders):
#                 print(f"{i}: {folder}")
#             choice = input()
#             if not choice.isdigit():
#                 choice = -1
#             else:
#                 choice = int(choice)
#         idx = choice

#     model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
#     model = YOLO(model_path)

#     # Directory with images
#     output_dir = this_dir / "predictions" # Replace with the directory where you want to save predictions
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Create images and labels subdirectories
#     images_output_dir = output_dir / 'images'
#     labels_output_dir = output_dir / 'labels'
#     images_output_dir.mkdir(parents=True, exist_ok=True)
#     labels_output_dir.mkdir(parents=True, exist_ok=True)

#     # Iterate through the images in the directory
#     for img_path in images_dir.glob('*'):
#         if img_path.suffix not in ['.png', '.jpg']:
#             continue
#         output_path_img = images_output_dir / img_path.name  # Save image in 'images' folder
#         output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
#         predict_and_save(model, img_path, output_path_img, output_path_txt)

#     print(f"Predicted images saved in {images_output_dir}")
#     print(f"Bounding box labels saved in {labels_output_dir}")
#     data = this_dir / 'yolo_params.yaml'
#     print(f"Model parameters saved in {data}")
#     metrics = model.val(data=data, split="test")

#!/usr/bin/env python3
import argparse
import os
import sys
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import matplotlib.patches as patches
import shutil
import cv2

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_model(model_path):
    """Load a trained YOLO model."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

def validate_data_yaml(data_yaml):
    """Validate the data YAML file and return the verified path."""
    try:
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check if test or val paths exist
        test_path = data_config.get('test', None)
        val_path = data_config.get('val', None)
        
        # Verify that at least one path exists
        if not test_path and not val_path:
            print(f"Warning: Neither 'test' nor 'val' paths found in {data_yaml}")
            return False
        
        # Validate the paths point to existing directories or files
        paths_to_check = []
        if test_path:
            paths_to_check.append(('test', test_path))
        if val_path:
            paths_to_check.append(('val', val_path))
        
        # Check if the paths exist
        for path_type, path in paths_to_check:
            # First check if it's a direct path
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Check if there's an 'images' subdirectory
                    images_dir = os.path.join(path, 'images')
                    if os.path.exists(images_dir):
                        print(f"Found {path_type} images directory at: {images_dir}")
                    else:
                        print(f"Warning: '{path_type}' path exists but no 'images' subdirectory found at {path}")
                else:
                    # It might be a text file listing image paths
                    print(f"Found {path_type} path file at: {path}")
            else:
                # Try to resolve relative paths
                for base_dir in [os.path.dirname(data_yaml), os.getcwd()]:
                    resolved_path = os.path.join(base_dir, path)
                    if os.path.exists(resolved_path):
                        # Update the path in the config
                        data_config[path_type] = resolved_path
                        if os.path.isdir(resolved_path):
                            images_dir = os.path.join(resolved_path, 'images')
                            if os.path.exists(images_dir):
                                print(f"Found {path_type} images directory at: {images_dir}")
                                break
                        else:
                            print(f"Found {path_type} path file at: {resolved_path}")
                            break
                else:
                    print(f"Error: Could not find {path_type} path at {path}")
                    print(f"Tried: {path} and {os.path.join(os.path.dirname(data_yaml), path)}")
                    return False
        
        # Save the validated config back to a temporary file
        temp_yaml = data_yaml + '.validated.yaml'
        with open(temp_yaml, 'w') as f:
            yaml.dump(data_config, f)
        
        return temp_yaml
    except Exception as e:
        print(f"Error validating data YAML: {e}")
        return False

def run_prediction(model, data_yaml, output_dir, conf_threshold=0.25, iou_threshold=0.7, device=0, use_tta=False, tta_types=None):
    """Run prediction on test data and save metrics."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories for results
    results_dir = os.path.join(output_dir, f"predictions_{timestamp}")
    images_dir = os.path.join(results_dir, "images")
    create_directory_if_not_exists(results_dir)
    create_directory_if_not_exists(images_dir)
    
    # Validate the data YAML file
    validated_yaml = validate_data_yaml(data_yaml)
    if not validated_yaml:
        print("Warning: Proceeding with original data YAML, but prediction may fail.")
        validated_yaml = data_yaml
    
    try:
        # Save information about TTA usage
        tta_info = {
            "use_tta": use_tta,
            "tta_types": tta_types if tta_types else "Default"
        }
        with open(os.path.join(results_dir, "tta_config.json"), 'w') as f:
            json.dump(tta_info, f, indent=4)
        
        # Run validation on test data with TTA if enabled
        results = model.val(
            data=validated_yaml,
            split="test",
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            save_json=True,
            save_txt=True,
            save_conf=True,
            project=results_dir,
            name="test_results",
            augment=use_tta  # This enables default TTA in YOLOv8
        )
    
        # Extract and save metrics
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        
        # Add mAP values to metrics
        if hasattr(results, 'metrics'):
            for k, v in results.metrics.items():
                metrics[k] = float(v) if isinstance(v, (int, float, np.number)) else v
        
        # Add TTA information to metrics
        metrics['test_time_augmentation'] = use_tta
        if use_tta and tta_types:
            metrics['tta_types'] = tta_types
        
        # Save metrics to JSON
        metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")
        
        return results, results_dir
    
    except Exception as e:
        raise Exception(f"Something went wrong in prediction: {e}")

def visualize_predictions(model, data_yaml, results_dir, num_images=10, conf_threshold=0.25, use_tta=False, tta_types=None):
    """Generate visualization of predictions on sample test images with optional TTA."""
    # Load the data configuration
    try:
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get test path
        if 'test' in data_config:
            test_path = data_config['test']
        elif 'val' in data_config:
            test_path = data_config['val']
        else:
            print("Warning: Could not find test or validation data path in YAML file")
            return
        
        # Find class names
        class_names = data_config.get('names', {})
        
    except Exception as e:
        print(f"Error loading data configuration: {e}")
        return
    
    # Get list of test images
    import glob
    test_images = []
    
    # First try to locate the images directory
    if os.path.isdir(test_path):
        # Check if there's an 'images' subdirectory
        images_dir = os.path.join(test_path, "images")
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            test_path = images_dir
        
        # Look for image files
        print(f"Looking for images in: {test_path}")
        test_images = glob.glob(os.path.join(test_path, "**/*.jpg"), recursive=True)
        test_images += glob.glob(os.path.join(test_path, "**/*.jpeg"), recursive=True)
        test_images += glob.glob(os.path.join(test_path, "**/*.png"), recursive=True)
        
        if not test_images:
            print(f"Warning: No images found in {test_path}")
            # Try searching one level deeper
            deeper_images_dir = os.path.join(test_path, "**")
            test_images = glob.glob(os.path.join(deeper_images_dir, "*.jpg"), recursive=True)
            test_images += glob.glob(os.path.join(deeper_images_dir, "*.jpeg"), recursive=True)
            test_images += glob.glob(os.path.join(deeper_images_dir, "*.png"), recursive=True)
            
            if test_images:
                print(f"Found {len(test_images)} images in subdirectories")
    elif os.path.isfile(test_path):
        # It might be a text file with image paths
        try:
            with open(test_path, 'r') as f:
                test_images = [line.strip() for line in f.readlines()]
            print(f"Found {len(test_images)} image paths in {test_path}")
        except Exception as e:
            print(f"Error reading image paths from {test_path}: {e}")
    else:
        print(f"Warning: Test path {test_path} does not exist or is not accessible")
        
    if not test_images:
        print("No test images found. Cannot generate visualizations.")
        return
    
    # Select a subset of images for visualization
    if len(test_images) > num_images:
        test_images = np.random.choice(test_images, num_images, replace=False)
    
    # Create a directory for visualizations
    vis_dir = os.path.join(results_dir, "visualizations")
    create_directory_if_not_exists(vis_dir)
    
    # Create subdirectories to compare with/without TTA
    if use_tta:
        no_tta_dir = os.path.join(vis_dir, "no_tta")
        tta_dir = os.path.join(vis_dir, "with_tta")
        create_directory_if_not_exists(no_tta_dir)
        create_directory_if_not_exists(tta_dir)
    
    # Run predictions on test images and create visualizations
    for img_path in test_images:
        try:
            # If this is a path from a file list, get the actual path
            if not os.path.exists(img_path) and os.path.exists(img_path.lstrip()):
                img_path = img_path.lstrip()
            
            # Get the base image name
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            if use_tta:
                # First predict without TTA
                results_no_tta = model.predict(img_path, conf=conf_threshold, save=True, project=no_tta_dir)
                
                # Then predict with TTA
                results_tta = model.predict(img_path, conf=conf_threshold, save=True, project=tta_dir, augment=True)
                
                # Create side-by-side comparison
                create_comparison_visualization(img_path, results_no_tta, results_tta, 
                                               os.path.join(vis_dir, f"{base_name}_comparison.jpg"), 
                                               class_names)
            else:
                # Just predict normally
                results = model.predict(img_path, conf=conf_threshold, save=True, project=vis_dir)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Visualizations saved to: {vis_dir}")

def create_comparison_visualization(img_path, results_no_tta, results_tta, output_path, class_names):
    """Create a side-by-side comparison of predictions with and without TTA."""
    try:
        # Load the original image
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Could not load image {img_path}")
            return
            
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Get the first result from each prediction (assuming single image input)
        result_no_tta = results_no_tta[0] if results_no_tta else None
        result_tta = results_tta[0] if results_tta else None
        
        # Create a figure for comparison
        plt.figure(figsize=(18, 9))
        
        # Plot image without TTA
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Without Test Time Augmentation")
        
        # Draw bounding boxes for no TTA
        if result_no_tta and hasattr(result_no_tta, 'boxes'):
            boxes = result_no_tta.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Get class name
                cls_name = class_names.get(cls, f"Class {cls}")
                
                # Create rectangle patch
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                        edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label
                plt.text(x1, y1-5, f"{cls_name}: {conf:.2f}", color='white', 
                         backgroundcolor='red', fontsize=8)
        
        # Plot image with TTA
        plt.subplot(1, 2, 2)
        plt.imshow(original_img)
        plt.title("With Test Time Augmentation")
        
        # Draw bounding boxes for TTA
        if result_tta and hasattr(result_tta, 'boxes'):
            boxes = result_tta.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Get class name
                cls_name = class_names.get(cls, f"Class {cls}")
                
                # Create rectangle patch
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                        edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label
                plt.text(x1, y1-5, f"{cls_name}: {conf:.2f}", color='white', 
                         backgroundcolor='green', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")

def generate_report(results, model_path, results_dir, use_tta=False, tta_types=None):
    """Generate a summary report of prediction results with TTA information."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract metrics for reporting
    metrics = {}
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
    elif hasattr(results, 'metrics'):
        metrics = results.metrics
    
    # Format metrics for display
    metrics_formatted = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.number)):
            metrics_formatted[k] = round(float(v), 4)
        else:
            metrics_formatted[k] = v
    
    # Extract key metrics
    map50 = metrics_formatted.get('metrics/mAP50(B)', metrics_formatted.get('map50', 'N/A'))
    map50_95 = metrics_formatted.get('metrics/mAP50-95(B)', metrics_formatted.get('map', 'N/A'))
    precision = metrics_formatted.get('metrics/precision(B)', metrics_formatted.get('precision', 'N/A'))
    recall = metrics_formatted.get('metrics/recall(B)', metrics_formatted.get('recall', 'N/A'))
    
    # Create a markdown report
    report_file = os.path.join(results_dir, f"prediction_report_{timestamp}.md")
    with open(report_file, 'w') as f:
        f.write(f"# YOLOv8 Prediction Report\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Normalize and split the path
        parts = os.path.normpath(model_path).split(os.sep)

        # Find the index of "results" in the path
        try:
            idx = parts.index("results")
            relative_path = os.path.join(*parts[idx:])
            print(relative_path)
        except ValueError:
            print("'results' not found in path")
        f.write(f"**Model:**  {relative_path}\n\n")
        
        # Add TTA information
        f.write("## Test Time Augmentation\n\n")
        f.write(f"* **TTA Enabled:** {use_tta}\n")
        if use_tta and tta_types:
            f.write(f"* **TTA Types:** {', '.join(tta_types)}\n")
        elif use_tta:
            f.write(f"* **TTA Types:** Default YOLOv8 augmentations (flips, scales)\n")
        f.write("\n")
        
        f.write("## Key Metrics\n\n")
        f.write(f"* **mAP50:** {map50}\n")
        f.write(f"* **mAP50-95:** {map50_95}\n")
        f.write(f"* **Precision:** {precision}\n")
        f.write(f"* **Recall:** {recall}\n\n")
        
        f.write("## All Metrics\n\n")
        for k, v in metrics_formatted.items():
            f.write(f"* **{k}:** {v}\n")
        
        f.write("\n## Directories\n\n")
        f.write(f"* Prediction results: `{results_dir}`\n")
        f.write(f"* Visualizations: `{os.path.join(results_dir, 'visualizations')}`\n")
    
    print(f"Report generated at: {report_file}")
    
    # Also save as JSON for programmatic access
    with open(os.path.join(results_dir, f"prediction_report_{timestamp}.json"), 'w') as f:
        json.dump({
            "model": os.path.basename(model_path),
            "timestamp": timestamp,
            "test_time_augmentation": {
                "enabled": use_tta,
                "types": tta_types if tta_types else "Default YOLOv8 augmentations"
            },
            
            "metrics": metrics_formatted,
            "key_metrics": {
                "mAP50": map50,
                "mAP50-95": map50_95,
                "precision": precision,
                "recall": recall
            },
            "directories": {
                "results": results_dir,
                "visualizations": os.path.join(results_dir, 'visualizations')
            }
        }, f, indent=4)

def plot_confusion_matrix(results_dir):
    """Generate confusion matrix plot if available in the results."""
    conf_matrix_path = os.path.join(results_dir, "test_results", "confusion_matrix.png")
    if os.path.exists(conf_matrix_path):
        # Copy to main results directory for easier access
        shutil.copy(conf_matrix_path, os.path.join(results_dir, "confusion_matrix.png"))
        print(f"Confusion matrix saved to: {os.path.join(results_dir, 'confusion_matrix.png')}")
    else:
        print("Confusion matrix not found in results")

def find_latest_model(results_dir):
    """Find the latest best model in the results directory."""
    weights_dir = os.path.join(results_dir, "weights")
    if not os.path.exists(weights_dir):
        weights_dir = os.path.join(os.path.dirname(results_dir), "weights")
        if not os.path.exists(weights_dir):
            return None
    
    # Look for models with "best" in the name
    best_models = []
    for root, _, files in os.walk(weights_dir):
        for file in files:
            if file.endswith(".pt") and "best" in file.lower():
                best_models.append(os.path.join(root, file))
    
    # Sort by modification time and get the latest
    if best_models:
        best_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return best_models[0]
    return None

def find_model_by_run_name(results_dir, run_name):
    """Find a model by run name in the results directory."""
    # Check if the run_name contains a full path to a model file
    if os.path.exists(run_name) and run_name.endswith(".pt"):
        return run_name
    
    # Look for the run in results directory
    run_dir = os.path.join(results_dir, run_name)
    if not os.path.exists(run_dir):
        run_dir = os.path.join(results_dir, "tuning_run_" + run_name.lstrip("tuning_run_"))
    
    if os.path.exists(run_dir):
        weights_dir = os.path.join(run_dir, "weights")
        if os.path.exists(weights_dir):
            best_model = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_model):
                return best_model
    
    # If we get here, we didn't find the model
    print(f"Warning: Could not find model for run {run_name}. Looking for latest model...")
    return find_latest_model(results_dir)

def get_tta_types_from_string(tta_types_str):
    """Parse the TTA types string into a list of augmentation types."""
    if not tta_types_str:
        return None
        
    # Split the string by commas
    return [t.strip().lower() for t in tta_types_str.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run predictions with a trained YOLOv8 model")
    
    # Required arguments
    parser.add_argument('--model', type=str, required=False,
                        help='Path to the model weights file or run name')
    parser.add_argument('--data', type=str, default="yolo_params.yaml",
                        help='Path to the data YAML file')
    
    # Optional arguments
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to run on (0 for first GPU)')
    parser.add_argument('--output', type=str, default="prediction_results",
                        help='Directory to save prediction results')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of test images to visualize')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data directory, overrides data YAML test path')
    
    # TTA-specific arguments
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test Time Augmentation for predictions')
    parser.add_argument('--tta_types', type=str, default=None,
                        help='Comma-separated list of TTA types to use (e.g., "flip,scale,rotate")')
    parser.add_argument('--compare_tta', action='store_true',
                        help='Generate side-by-side visualizations of predictions with and without TTA')
    
    args = parser.parse_args()
    
    # Get the current directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    
    # Create output directory
    output_dir = os.path.join(this_dir, args.output)
    create_directory_if_not_exists(output_dir)
    
    # Find the model file
    model_path = None
    if args.model:
        # Check if it's a direct path
        if os.path.exists(args.model):
            model_path = args.model
        else:
            # Try to find the model by run name
            results_dir = os.path.join(this_dir, "results")
            model_path = find_model_by_run_name(results_dir, args.model)
    else:
        # Find the latest best model
        results_dir = os.path.join(this_dir, "results")
        model_path = find_latest_model(results_dir)
    
    if not model_path:
        print("Error: Could not find a valid model file. Please specify a valid model path or run name.")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Resolve data path
    if not os.path.exists(args.data):
        data_path = os.path.join(this_dir, args.data)
        if not os.path.exists(data_path):
            print(f"Error: Data configuration not found at {args.data}")
            sys.exit(1)
    else:
        data_path = args.data
    
    # If test data path is provided, modify the YAML file
    if args.test_data:
        try:
            import yaml
            # Load the existing YAML
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Update test path
            original_test = data_config.get('test', None)
            original_val = data_config.get('val', None)
            data_config['test'] = args.test_data
            
            # Save to a new YAML file
            modified_data_path = os.path.join(output_dir, "modified_data.yaml")
            with open(modified_data_path, 'w') as f:
                yaml.dump(data_config, f)
            
            print(f"Modified data YAML with test path: {args.test_data}")
            print(f"Original test path: {original_test}")
            print(f"Original val path: {original_val}")
            
            data_path = modified_data_path
        except Exception as e:
            print(f"Error modifying data YAML: {e}")
    
    # Process TTA types
    tta_types = get_tta_types_from_string(args.tta_types) if args.tta else None
    
    # Print TTA configuration
    if args.tta:
        print(f"Test Time Augmentation enabled")
        if tta_types:
            print(f"TTA types: {', '.join(tta_types)}")
        else:
            print("Using default YOLOv8 TTA (flips and scales)")
    
    # Load the model
    model = load_model(model_path)
    
    # Run prediction
    results, results_dir = run_prediction(
        model=model,
        data_yaml=data_path,
        output_dir=output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        use_tta=args.tta,
        tta_types=tta_types
    )
    
    # Generate visualizations
    visualize_predictions(
        model=model,
        data_yaml=data_path,
        results_dir=results_dir,
        num_images=args.num_vis,
        conf_threshold=args.conf,
        use_tta=args.compare_tta,  # Only create comparison visuals if requested
        tta_types=tta_types
    )
    
    # Generate confusion matrix plot
    plot_confusion_matrix(results_dir)
    
    # Generate report
    generate_report(results, model_path, results_dir, args.tta, tta_types)
    
    print(f"\nPrediction complete! Results saved to: {results_dir}")