import argparse
from ultralytics import YOLO
import os
import sys
import datetime
import json

EPOCHS = 1
MOSAIC = 0.1
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()
    
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Create directories for logs and weights
    logs_dir = os.path.join(this_dir, "results", "logs")
    weights_dir = os.path.join(this_dir, "results", "weights")
    create_directory_if_not_exists(logs_dir)
    create_directory_if_not_exists(weights_dir)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=0,
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        project=os.path.join(this_dir, "results"),  # Set project directory to results
        name="train"  # This creates a subdirectory under project
    )
    
    # Get mAP50 from training results
    if hasattr(results, 'results_dict') and 'metrics/mAP50(B)' in results.results_dict:
        map50 = results.results_dict['metrics/mAP50(B)']
    else:
        # Try to get mAP50 from the last epoch's results
        try:
            # Format mAP50 to have 4 decimal places
            map50 = round(results.metrics.get('map50', 0), 4)
        except (AttributeError, KeyError):
            # If unable to get mAP50, use a placeholder
            map50 = "unknown"
    
    # Create filenames with timestamp and mAP50
    base_filename = f"{timestamp}_mAP50_{map50}"
    
    # Save logs (metrics and hyperparameters)
    logs_filename = os.path.join(logs_dir, f"{base_filename}_logs.json")
    logs_data = {
        "timestamp": timestamp,
        "metrics": results.metrics if hasattr(results, 'metrics') else {},
        "hyperparameters": {
            "epochs": args.epochs,
            "mosaic": args.mosaic,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "single_cls": args.single_cls
        }
    }
    
    with open(logs_filename, 'w') as f:
        json.dump(logs_data, f, indent=4)
    print(f"Training logs saved to: {logs_filename}")
    
    # Save trained weights
    # The model is already saved by YOLO in results/train/weights/
    # We'll copy/rename the best model to our weights directory
    best_model_path = os.path.join(this_dir, "results", "train", "weights", "best.pt")
    if os.path.exists(best_model_path):
        import shutil
        weights_filename = os.path.join(weights_dir, f"{base_filename}_best.pt")
        shutil.copy(best_model_path, weights_filename)
        print(f"Trained weights saved to: {weights_filename}")
    else:
        print("Warning: Best model weights not found!")
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''