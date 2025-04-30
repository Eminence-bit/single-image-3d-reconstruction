import os
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

class MiDaSDepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        """
        Initialize MiDaS depth estimator
        model_type options: 
        - MiDaS v3.1: "DPT_Large" (best performance but slower), 
                      "DPT_Hybrid" (balanced), 
                      "MiDaS_small" (fastest)
        """
        print(f"Initializing MiDaS with model type: {model_type}")
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model configuration
        self.model_type = model_type
        
        # Load model directly from the torch hub
        print("Loading MiDaS model from torch hub...")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        print("Loading MiDaS transforms...")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        print("MiDaS initialization complete!")

    def estimate_depth(self, img_path):
        """
        Estimate depth from image
        """
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply input transform
        input_batch = self.transform(img).to(self.device)
        
        # Prediction
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return depth_map, img
    
    def generate_colored_depth_map(self, depth_map):
        """
        Generate a colored depth map using a jet colormap
        """
        colored_depth_map = plt.cm.jet(depth_map)[:, :, :3]  # Get RGB channels only
        colored_depth_map = (colored_depth_map * 255).astype(np.uint8)
        return colored_depth_map
    
    def process_image(self, img_path, output_dir="data/output"):
        """
        Process a single image and generate depth map
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Estimate depth
        depth_map, original_img = self.estimate_depth(img_path)
        
        # Create colored depth map
        colored_depth_map = self.generate_colored_depth_map(depth_map)
        
        # Save results
        image_name = Path(img_path).stem
        depth_map_path = f"{output_dir}/{image_name}_depth.jpg"
        colored_depth_map_path = f"{output_dir}/{image_name}_depth_colored.jpg"
        
        # Save grayscale depth map
        cv2.imwrite(depth_map_path, (depth_map * 255).astype(np.uint8))
        
        # Save colored depth map
        cv2.imwrite(colored_depth_map_path, cv2.cvtColor(colored_depth_map, cv2.COLOR_RGB2BGR))
        
        print(f"Depth map generated for {img_path}")
        print(f"Saved to {depth_map_path} and {colored_depth_map_path}")
        
        return depth_map_path, colored_depth_map_path
    
    def process_directory(self, input_dir="data/output", pattern="_masks.jpg", output_dir="data/output/depth"):
        """
        Process all mask images in the input directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed_files = []
        
        for img_file in os.listdir(input_dir):
            if pattern in img_file and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, img_file)
                print(f"Processing {img_path}...")
                depth_map_path, colored_depth_map_path = self.process_image(img_path, output_dir)
                processed_files.append((depth_map_path, colored_depth_map_path))
        
        return processed_files
    
    def process_segmented_images(self, input_dir="data/output", output_dir="data/output/depth"):
        """
        Process all segmented images (both mask and segmented versions)
        """
        return self.process_directory(input_dir, "_masks.jpg", output_dir)
    
    def process_all_detected_images(self, input_dir="data/output", output_dir="data/output/depth"):
        """
        Process all detected images (yolo, segmented, and detected)
        """
        patterns = ["_yolo.jpg", "_segmented.jpg", "_detected.jpg"]
        all_processed = []
        
        for pattern in patterns:
            print(f"Processing images with pattern: {pattern}")
            processed = self.process_directory(input_dir, pattern, output_dir)
            all_processed.extend(processed)
        
        return all_processed

if __name__ == "__main__":
    # Create depth estimator with DPT_Large model (best quality)
    # Other options: "DPT_Hybrid" (balanced) or "MiDaS_small" (faster)
    depth_estimator = MiDaSDepthEstimator(model_type="DPT_Large")
    
    # Create output directory for depth maps
    depth_output_dir = "data/output/depth/large"
    os.makedirs(depth_output_dir, exist_ok=True)
    
    # Process all mask images
    depth_estimator.process_segmented_images(output_dir=depth_output_dir)
    
    # Uncomment to process all detected images (yolo, segmented, detected)
    # depth_estimator.process_all_detected_images(output_dir=depth_output_dir)