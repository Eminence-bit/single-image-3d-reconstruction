import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

class ObjectDetector:
    def __init__(self):
        # Load pre-trained model - this is equivalent to Detectron2's faster_rcnn models
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
        self.model.eval()
        
        # Get class names from the weights
        self.classes = weights.meta["categories"]
        self.transform = weights.transforms()
        
    def detect(self, image_path):
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Get predictions
        pred_boxes = predictions[0]["boxes"].cpu().numpy()
        pred_scores = predictions[0]["scores"].cpu().numpy()
        pred_labels = predictions[0]["labels"].cpu().numpy()
        
        # Convert back to a format suitable for visualization
        image_np = np.array(image)
        
        # Draw boxes and labels
        for box, score, label_idx in zip(pred_boxes, pred_scores, pred_labels):
            if score > 0.5:  # Only show confident predictions
                x1, y1, x2, y2 = map(int, box)
                label = self.classes[label_idx]
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{label}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_np, pred_boxes, pred_scores, pred_labels
    
    def visualize_and_save(self, image_path, output_dir="data/output"):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect objects
        result_image, boxes, scores, labels = self.detect(image_path)
        
        # Save the result
        image_name = Path(image_path).stem
        output_path = f"{output_dir}/{image_name}_detected.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Print detection results
        for box, score, label_idx in zip(boxes, scores, labels):
            if score > 0.5:
                print(f"Detected {self.classes[label_idx]} with confidence {score:.2f}")
        
        print(f"Result saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Initialize the detector
    detector = ObjectDetector()
    
    # Process all images in the input directory
    input_dir = "data/input"
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_file)
            print(f"Processing {img_path}...")
            detector.visualize_and_save(img_path)