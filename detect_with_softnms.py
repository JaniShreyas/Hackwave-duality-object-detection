from ultralytics import YOLO
import os
import cv2
from soft_nms import integrate_soft_nms

def run_inference(model_path, image_dir, output_dir):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            results = model(img_path)
            
            for r in results:
                im = r.plot()
                output_path = os.path.join(output_dir, img_file)
                cv2.imwrite(output_path, im)
                print(f"{output_dir} - {img_file}: Found {len(r.boxes)} objects")
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    print(f"  Class: {cls}, Confidence: {conf:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

def main():
    model_path = "C:/Users/drish/OneDrive/Desktop/CS/Duality_HackWave/Hackwave-duality-object-detection/weights/weights_run_4/weights/best.pt"
    image_dir = "C:/Users/drish/Downloads/Hackathon_Dataset/HackByte_Dataset/data/test/images"

    # ------- Run with Soft-NMS -------
    print("Running with Soft-NMS...")
    integrate_soft_nms()
    run_inference(model_path, image_dir, "runs/detect/soft_nms_results")

    # ------- Run with Original NMS -------
    print("\nRunning with Original NMS...")
    # IMPORTANT: Reload Python's default NMS
    from importlib import reload
    import ultralytics.utils.ops
    reload(ultralytics.utils.ops)  # This restores the original NMS
    run_inference(model_path, image_dir, "runs/detect/original_nms_results")

if __name__ == "__main__":
    main()

