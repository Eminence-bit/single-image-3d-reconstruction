import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import trimesh
from skimage import measure
from scipy import ndimage
from ultralytics import YOLO
from view_synthesizer import ViewSynthesizer

class Pipeline3D:
    def __init__(self, input_dir="data/input", output_dir="data/output"):
        """
        Initialize the 3D reconstruction pipeline
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, "depth")
        self.voxels_dir = os.path.join(output_dir, "voxels")
        self.meshes_dir = os.path.join(output_dir, "meshes")
        self.views_dir = os.path.join(output_dir, "views")  # New directory for synthetic views
        
        # Create output directories
        for directory in [output_dir, self.depth_dir, self.voxels_dir, self.meshes_dir, self.views_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize view synthesizer
        self.view_synthesizer = ViewSynthesizer(device=self.device)
    
    def segment_objects(self, img_path, method="maskrcnn"):
        """
        Segment objects in an image using Mask R-CNN or YOLO
        
        Args:
            img_path: Path to the input image
            method: Segmentation method ('maskrcnn' or 'yolo')
            
        Returns:
            Dict with paths to output files
        """
        img_name = Path(img_path).stem
        
        if method == "maskrcnn":
            return self._segment_with_maskrcnn(img_path, img_name)
        elif method == "yolo":
            return self._segment_with_yolo(img_path, img_name)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _segment_with_maskrcnn(self, img_path, img_name):
        """
        Segment objects using Mask R-CNN
        """
        print(f"Segmenting objects in {img_path} using Mask R-CNN...")
        
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        
        # Load Mask R-CNN model
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.to(self.device)
        model.eval()
        
        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor])[0]
        
        # Extract masks, boxes, scores, labels
        masks = prediction["masks"].cpu().numpy()
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        
        # Create masked image (for all objects)
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        segmented_image = image.copy()
        
        conf_threshold = 0.5
        for i, score in enumerate(scores):
            if score > conf_threshold:
                mask = masks[i, 0]
                mask_binary = (mask > 0.5).astype(np.uint8)
                mask_combined = np.maximum(mask_combined, mask_binary)
                
                # Add colored mask to segmented image
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                segmented_image[mask_binary > 0] = segmented_image[mask_binary > 0] * 0.7 + color * 0.3
                
                # Draw bounding box
                box = boxes[i].astype(np.int32)
                cv2.rectangle(segmented_image, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
                
                # Add label
                label_id = labels[i]
                label_text = f"Class {label_id}: {score:.2f}"
                cv2.putText(segmented_image, label_text, (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        # Save outputs
        detected_path = os.path.join(self.output_dir, f"{img_name}_detected.jpg")
        mask_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
        segmented_path = os.path.join(self.output_dir, f"{img_name}_segmented.jpg")
        
        # Save detected image with bounding boxes
        cv2.imwrite(detected_path, segmented_image)
        
        # Save mask
        cv2.imwrite(mask_path, mask_combined * 255)
        
        # Save segmented image (original with colored masks)
        segmented_vis = image.copy()
        segmented_vis[mask_combined > 0, 2] = 255  # Highlight in red
        cv2.imwrite(segmented_path, segmented_vis)
        
        print(f"Segmentation results saved to {self.output_dir}")
        return {
            "detected": detected_path,
            "masks": mask_path,
            "segmented": segmented_path
        }
    
    def _segment_with_yolo(self, img_path, img_name):
        """
        Segment objects using YOLOv8
        """
        print(f"Detecting objects in {img_path} using YOLOv8...")
        
        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Use nano model by default
        
        # Run inference
        results = model(img_path, conf=0.25)
        
        # Get the first result
        result = results[0]
        
        # Load the original image to draw on
        image = cv2.imread(img_path)
        
        # Process results
        detected_objects = []
        
        # Create empty mask for the detected objects
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for box in result.boxes:
            # Get box coordinates, confidence and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Create a simple mask from the bounding box (since YOLOv8 doesn't provide masks)
            cv2.rectangle(mask_combined, (x1, y1), (x2, y2), 255, -1)  # Filled rectangle
            
            detected_objects.append({
                'class': cls_name,
                'confidence': conf,
                'box': (x1, y1, x2, y2)
            })
        
        # Save results
        detected_path = os.path.join(self.output_dir, f"{img_name}_yolo.jpg")
        mask_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
        
        cv2.imwrite(detected_path, image)
        cv2.imwrite(mask_path, mask_combined)
        
        print(f"YOLO detected {len(detected_objects)} objects in {img_path}")
        print(f"Results saved to {detected_path}")
        
        return {
            "detected": detected_path,
            "masks": mask_path
        }
    
    def generate_depth_map(self, img_path, model_type="DPT_Large"):
        """
        Generate depth map from an image using MiDaS
        
        Args:
            img_path: Path to the input image (usually a mask)
            model_type: MiDaS model type ('DPT_Large', 'DPT_Hybrid', or 'MiDaS_small')
            
        Returns:
            Dict with paths to output files
        """
        print(f"Generating depth map for {img_path} using MiDaS {model_type}...")
        
        img_name = Path(img_path).stem
        
        # Initialize MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(self.device)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply input transform
        input_batch = transform(img).to(self.device)
        
        # Prediction
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Create colored depth map
        colored_depth_map = plt.cm.jet(depth_map)[:, :, :3]
        colored_depth_map = (colored_depth_map * 255).astype(np.uint8)
        
        # Save results
        depth_path = os.path.join(self.depth_dir, f"{img_name}_depth.jpg")
        colored_depth_path = os.path.join(self.depth_dir, f"{img_name}_depth_colored.jpg")
        
        # Save grayscale depth map
        cv2.imwrite(depth_path, (depth_map * 255).astype(np.uint8))
        
        # Save colored depth map
        cv2.imwrite(colored_depth_path, cv2.cvtColor(colored_depth_map, cv2.COLOR_RGB2BGR))
        
        print(f"Depth map generated for {img_path}")
        print(f"Saved to {depth_path} and {colored_depth_path}")
        
        return {
            "depth": depth_path,
            "colored_depth": colored_depth_path,
            "depth_data": depth_map
        }
    
    def generate_synthetic_views(self, image_path, depth_path, mask_path, num_views=8):
        """
        Generate synthetic views of the object from different angles
        
        Args:
            image_path: Path to the original image
            depth_path: Path to the depth map
            mask_path: Path to the object mask
            num_views: Number of views to generate
            
        Returns:
            Dictionary with paths to synthetic views
        """
        print(f"\n--- Generating Synthetic Views ({num_views} views) ---")
        
        img_name = Path(image_path).stem
        output_dir = os.path.join(self.views_dir, img_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Generate synthetic views
        synthetic_views = self.view_synthesizer.generate_synthetic_views(
            image, depth, mask, num_views=num_views
        )
        
        # Save views
        output_paths = self.view_synthesizer.save_synthetic_views(
            synthetic_views, output_dir, img_name
        )
        
        print(f"Generated {len(synthetic_views)} synthetic views from {image_path}")
        
        return output_paths
    
    def create_multi_view_voxel_grid(self, original_image_path, original_depth_path, mask_path, synthetic_views, resolution=128):
        """
        Create a voxel grid using multiple views for better 3D reconstruction
        
        Args:
            original_image_path: Path to the original image
            original_depth_path: Path to the original depth map
            mask_path: Path to the object mask
            synthetic_views: Dictionary with paths to synthetic views
            resolution: Voxel grid resolution
            
        Returns:
            3D voxel grid as numpy array
        """
        print(f"\n--- Creating Multi-View Voxel Grid ---")
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # Load original depth map
        original_depth = self.load_depth_map(original_depth_path)
        
        # Create initial voxel grid from the original view
        voxel_grid = self.create_voxel_grid(original_depth, mask, resolution)
        
        # Integrate information from synthetic views
        for i, depth_path in enumerate(synthetic_views['depth']):
            print(f"Integrating synthetic view {i+1}/{len(synthetic_views['depth'])}...")
            
            # Load synthetic depth map
            synthetic_depth = self.load_depth_map(depth_path)
            
            # Create voxel grid for this view
            view_grid = self.create_voxel_grid(synthetic_depth, mask, resolution)
            
            # Combine with the main grid (using maximum values to ensure complete geometry)
            voxel_grid = np.maximum(voxel_grid, view_grid)
        
        # Apply 3D smoothing to clean up noise introduced by multiple views
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.8)
        
        return voxel_grid
    
    def load_depth_map(self, path):
        """
        Load a depth map and normalize it
        
        Args:
            path: Path to the depth map image
            
        Returns:
            Normalized depth map as numpy array
        """
        depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Normalize to [0, 1]
        if depth.max() > 1.0:
            depth = depth.astype(np.float32) / 255.0
        return depth
    
    def create_voxel_grid(self, depth_map, mask=None, resolution=128):
        """
        Create a 3D voxel grid from a depth map
        
        Args:
            depth_map: Depth map as a numpy array
            mask: Optional binary mask
            resolution: Voxel grid resolution
            
        Returns:
            3D voxel grid as numpy array
        """
        print(f"Creating voxel grid with resolution {resolution}...")
        
        # If no mask is provided, create a simple one
        if mask is None:
            mask = np.ones_like(depth_map)
            
        # Normalize depth map to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / 255.0
            
        # Apply the mask
        masked_depth = depth_map * (mask > 0)
        
        # Clean depth map
        kernel = np.ones((3,3), np.uint8)
        cleaned_depth = cv2.morphologyEx(masked_depth.astype(np.float32), cv2.MORPH_OPEN, kernel)
        
        # Resize to match voxel resolution
        depth_resized = cv2.resize(cleaned_depth, (resolution, resolution))
        
        # Create 3D grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Use a more sophisticated approach to create the 3D volume
        for x in range(resolution):
            for y in range(resolution):
                depth_val = depth_resized[y, x]
                
                if depth_val > 0.01:
                    z_val = int(depth_val * resolution)
                    z_range = max(2, int(resolution * 0.05))
                    
                    z_start = max(0, z_val - z_range)
                    z_end = min(resolution - 1, z_val + z_range)
                    
                    voxel_grid[y, x, z_start:z_end+1] = 1.0
        
        # Apply 3D smoothing
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.8)
        
        return voxel_grid
    
    def create_mesh_from_voxels(self, voxel_grid, threshold=0.3):
        """
        Create a 3D mesh from a voxel grid using Marching Cubes
        
        Args:
            voxel_grid: 3D voxel grid as numpy array
            threshold: Isosurface threshold value
            
        Returns:
            Trimesh object
        """
        print(f"Creating mesh from voxel grid using Marching Cubes (threshold={threshold})...")
        
        # Pad the voxel grid to avoid boundary issues
        padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
        
        # Apply marching cubes
        vertices, faces, normals, _ = measure.marching_cubes(
            padded_grid, 
            level=threshold,
            allow_degenerate=False
        )
        
        # Adjust vertices to account for padding
        vertices = vertices - 1
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        # Apply Laplacian smoothing
        try:
            mesh = mesh.smoothed(method='laplacian', iterations=3)
        except Exception as e:
            print(f"Warning: Smoothing failed - {e}")
        
        return mesh
    
    def apply_texture_to_mesh(self, mesh, image_path, mask_path=None):
        """
        Apply texture from the original image to the mesh
        
        Args:
            mesh: Trimesh object
            image_path: Path to the original image for texturing
            mask_path: Optional path to the mask image
            
        Returns:
            Textured mesh object
        """
        print(f"Applying texture from {image_path} to mesh...")
        
        # Load the original image for texturing
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load texture image {image_path}. Using default coloring.")
            return self.apply_default_coloring(mesh)
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
        # Project texture onto mesh using mesh vertices positions
        # This is an improved approach for better texture mapping
        
        # First, establish the mapping from 3D to 2D
        # Normalize mesh vertices to [0, 1] range on XY plane
        vertices = mesh.vertices.copy()
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        range_vals = max_vals - min_vals
        
        # Prevent division by zero
        range_vals[range_vals == 0] = 1
        
        # Normalize vertices
        normalized_vertices = (vertices - min_vals) / range_vals
        
        # Get image dimensions
        img_height, img_width = image_rgb.shape[:2]
        
        # Improved mapping: use all dimensions to create better mapping
        # This helps when the object has complex 3D structure
        img_coords_x = (normalized_vertices[:, 0] * img_width * 0.8 + img_width * 0.1).astype(np.int32)
        img_coords_y = (normalized_vertices[:, 1] * img_height * 0.8 + img_height * 0.1).astype(np.int32)
        
        # Add some variation based on Z to create better texture wrapping
        z_offset_x = (normalized_vertices[:, 2] * 0.1 * img_width).astype(np.int32)
        z_offset_y = (normalized_vertices[:, 2] * 0.1 * img_height).astype(np.int32)
        
        img_coords_x = (img_coords_x + z_offset_x) % img_width
        img_coords_y = (img_coords_y + z_offset_y) % img_height
        
        # Clamp to image boundaries
        img_coords_x = np.clip(img_coords_x, 0, img_width - 1)
        img_coords_y = np.clip(img_coords_y, 0, img_height - 1)
        
        # Sample colors from image with improved color vibrancy
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        for i in range(len(vertices)):
            x, y = img_coords_x[i], img_coords_y[i]
            
            # Only apply texture where the mask is present (if mask is provided)
            if mask is None or mask[y, x] > 0:
                # Get color from the image and enhance it
                color = image_rgb[y, x].astype(np.float32)
                
                # Enhance color vibrancy (increase saturation)
                hsv = cv2.cvtColor(np.array([[color]]), cv2.COLOR_RGB2HSV)[0, 0]
                hsv[1] = min(hsv[1] * 1.2, 255)  # Increase saturation by 20%
                enhanced_color = cv2.cvtColor(np.array([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
                
                vertex_colors[i, :3] = enhanced_color
                vertex_colors[i, 3] = 255  # Full alpha channel
            else:
                # For points outside the mask, add a semi-transparent color
                vertex_colors[i, :3] = [200, 200, 200]  # Light gray
                vertex_colors[i, 3] = 128  # Semi-transparent
        
        # Create a new mesh with vertex colors
        textured_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_colors=vertex_colors
        )
        
        print("Texture mapping completed with enhanced color vibrancy.")
        
        return textured_mesh
    
    def apply_default_coloring(self, mesh):
        """
        Apply default coloring to the mesh based on depth
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Colored mesh object
        """
        # Get the vertices
        vertices = mesh.vertices
        
        # Determine min and max height (assuming Y is up)
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        height_range = y_max - y_min
        
        if height_range == 0:
            height_range = 1  # Prevent division by zero
        
        # Normalize heights to [0, 1]
        normalized_heights = (vertices[:, 1] - y_min) / height_range
        
        # Create a gradient coloring using jet colormap
        # Map values from [0, 1] to colors
        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        # Simple gradient from blue to red
        for i, h in enumerate(normalized_heights):
            if h < 0.25:
                # Blue to cyan
                b = 255
                g = int(255 * (h / 0.25))
                r = 0
            elif h < 0.5:
                # Cyan to green
                b = int(255 * (1 - (h - 0.25) / 0.25))
                g = 255
                r = 0
            elif h < 0.75:
                # Green to yellow
                b = 0
                g = 255
                r = int(255 * ((h - 0.5) / 0.25))
            else:
                # Yellow to red
                b = 0
                g = int(255 * (1 - (h - 0.75) / 0.25))
                r = 255
                
            colors[i] = [r, g, b, 255]
        
        # Create new mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_colors=colors
        )
        
        return colored_mesh
    
    def process_image(self, img_path, segmentation="maskrcnn", depth_model="DPT_Large", 
                     voxel_resolution=128, apply_texture=True, generate_views=True, num_views=8):
        """
        Process a single image through the entire pipeline with multi-view synthesis
        
        Args:
            img_path: Path to the input image
            segmentation: Segmentation method ('maskrcnn' or 'yolo')
            depth_model: MiDaS model type
            voxel_resolution: Voxel grid resolution
            apply_texture: Whether to apply texture to the mesh
            generate_views: Whether to generate synthetic views
            num_views: Number of synthetic views to generate
            
        Returns:
            Dict with paths to all output files
        """
        img_name = Path(img_path).stem
        
        # Step 1: Segment objects
        print(f"\n--- Step 1: Object Segmentation ({segmentation}) ---")
        segmentation_results = self.segment_objects(img_path, method=segmentation)
        
        # Step 2: Generate depth map from masks
        print(f"\n--- Step 2: Depth Map Generation ({depth_model}) ---")
        depth_results = self.generate_depth_map(segmentation_results["masks"], model_type=depth_model)
        
        # Step 3 (New): Generate synthetic views if requested
        synthetic_views = None
        if generate_views:
            synthetic_views = self.generate_synthetic_views(
                img_path,
                depth_results["depth"],
                segmentation_results["masks"],
                num_views=num_views
            )
        
        # Step 4: Create voxel grid (now with multi-view support)
        print(f"\n--- Step 4: Voxel Grid Creation (resolution={voxel_resolution}) ---")
        if generate_views and synthetic_views:
            # Use multi-view approach for better 3D reconstruction
            voxel_grid = self.create_multi_view_voxel_grid(
                img_path,
                depth_results["depth"],
                segmentation_results["masks"],
                synthetic_views,
                resolution=voxel_resolution
            )
        else:
            # Fallback to single-view approach
            mask = cv2.imread(segmentation_results["masks"], cv2.IMREAD_GRAYSCALE) / 255.0
            voxel_grid = self.create_voxel_grid(depth_results["depth_data"], mask, resolution=voxel_resolution)
        
        # Save voxel grid
        voxel_path = os.path.join(self.voxels_dir, f"{img_name}_voxels.npy")
        np.save(voxel_path, voxel_grid)
        
        # Visualize voxel grid
        self._visualize_voxel_grid(voxel_grid, img_name)
        
        # Step 5: Create mesh
        print(f"\n--- Step 5: Mesh Creation ---")
        mesh = self.create_mesh_from_voxels(voxel_grid)
        
        # Step 6: Apply texture if requested
        if apply_texture:
            print(f"\n--- Step 6: Texture Mapping ---")
            try:
                textured_mesh = self.apply_texture_to_mesh(
                    mesh,
                    img_path,  # Use original image for texture
                    segmentation_results["masks"]  # Use mask to isolate the object
                )
                
                # Save textured mesh
                obj_path = os.path.join(self.meshes_dir, f"{img_name}_textured.obj")
                mtl_path = os.path.join(self.meshes_dir, f"{img_name}_textured.mtl")
                textured_mesh.export(obj_path)
                
                # Also save a PLY file which better preserves vertex colors
                ply_path = os.path.join(self.meshes_dir, f"{img_name}_textured.ply")
                textured_mesh.export(ply_path)
                
                # Create mesh preview with texture
                preview_path = os.path.join(self.meshes_dir, f"{img_name}_textured_preview.png")
                self._create_mesh_preview(textured_mesh, preview_path)
                
                print(f"Textured mesh saved to {obj_path} and {ply_path}")
                
                # Also save the untextured mesh for comparison
                untextured_obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
                mesh.export(untextured_obj_path)
            except Exception as e:
                print(f"Error applying texture: {e}")
                print("Saving untextured mesh only")
                obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
                stl_path = os.path.join(self.meshes_dir, f"{img_name}.stl")
                mesh.export(obj_path)
                mesh.export(stl_path)
                
                # Create mesh preview without texture
                preview_path = os.path.join(self.meshes_dir, f"{img_name}_preview.png")
                self._create_mesh_preview(mesh, preview_path)
        else:
            # Save untextured mesh
            obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
            stl_path = os.path.join(self.meshes_dir, f"{img_name}.stl")
            mesh.export(obj_path)
            mesh.export(stl_path)
            
            # Create mesh preview
            preview_path = os.path.join(self.meshes_dir, f"{img_name}_preview.png")
            self._create_mesh_preview(mesh, preview_path)
        
        print(f"\n--- Processing complete for {img_path} ---")
        print(f"All results saved in {self.output_dir} and subdirectories")
        
        return {
            "segmentation": segmentation_results,
            "depth": depth_results,
            "views": synthetic_views if generate_views else None,
            "voxel": voxel_path,
            "mesh": {"obj": obj_path, "preview": preview_path}
        }
    
    def process_all_images(self, segmentation="maskrcnn", depth_model="DPT_Large", 
                          voxel_resolution=128, apply_texture=True, generate_views=True, num_views=8):
        """
        Process all images in the input directory
        
        Args:
            segmentation: Segmentation method
            depth_model: MiDaS model type
            voxel_resolution: Voxel grid resolution
            apply_texture: Whether to apply texture to the meshes
            generate_views: Whether to generate synthetic views
            num_views: Number of synthetic views to generate
        """
        image_paths = []
        
        # Find all images in the input directory
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(list(Path(self.input_dir).glob(f"*{ext}")))
        
        if not image_paths:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        
        results = {}
        for img_path in image_paths:
            print(f"\n=== Processing {img_path} ===")
            results[str(img_path)] = self.process_image(
                str(img_path),
                segmentation=segmentation,
                depth_model=depth_model,
                voxel_resolution=voxel_resolution,
                apply_texture=apply_texture,
                generate_views=generate_views,
                num_views=num_views
            )
        
        return results
    
    def _visualize_voxel_grid(self, voxel_grid, img_name):
        """
        Visualize a voxel grid with orthogonal slices
        """
        # Create a figure with 3 subplots for the 3 orthogonal slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Take central slices from each dimension
        mid_x = voxel_grid.shape[0] // 2
        mid_y = voxel_grid.shape[1] // 2
        mid_z = voxel_grid.shape[2] // 2
        
        # Create visualizations for each slice
        axes[0].imshow(voxel_grid[mid_x, :, :], cmap='jet')
        axes[0].set_title(f'YZ Slice (X={mid_x})')
        
        axes[1].imshow(voxel_grid[:, mid_y, :], cmap='jet')
        axes[1].set_title(f'XZ Slice (Y={mid_y})')
        
        axes[2].imshow(voxel_grid[:, :, mid_z], cmap='jet')
        axes[2].set_title(f'XY Slice (Z={mid_z})')
        
        # Save the figure
        viz_path = os.path.join(self.voxels_dir, f"{img_name}_voxel_slices.png")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved voxel visualization to {viz_path}")
    
    def _create_mesh_preview(self, mesh, output_path):
        """
        Create a preview image of a mesh
        """
        try:
            # Try to render with trimesh's built-in renderer
            scene = trimesh.Scene(mesh)
            png = scene.save_image(resolution=[1024, 768], visible=True)
            with open(output_path, 'wb') as f:
                f.write(png)
            print(f"Saved mesh preview: {output_path}")
        except Exception as e:
            print(f"Error with trimesh renderer: {e}")
            # Fall back to matplotlib
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(
                    mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    mesh.vertices[:, 2], 
                    triangles=mesh.faces,
                    cmap='viridis', 
                    alpha=0.8
                )
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"Saved mesh preview using matplotlib: {output_path}")
            except Exception as e2:
                print(f"Failed to create mesh preview: {e2}")


def main():
    """
    Main entry point for the 3D reconstruction pipeline
    """
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--input", default="data/input", help="Input directory containing images")
    parser.add_argument("--output", default="data/output", help="Output directory for results")
    parser.add_argument("--segmentation", choices=["maskrcnn", "yolo"], default="maskrcnn", 
                       help="Segmentation method to use (maskrcnn or yolo)")
    parser.add_argument("--depth-model", choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], 
                       default="DPT_Large", help="MiDaS depth model to use")
    parser.add_argument("--resolution", type=int, default=128, 
                       help="Voxel grid resolution (higher = more detail but slower)")
    parser.add_argument("--single-image", default=None, 
                       help="Process only a single image (provide path)")
    parser.add_argument("--no-texture", action="store_true",
                       help="Disable texture mapping (creates untextured meshes)")
    parser.add_argument("--no-views", action="store_true",
                       help="Disable synthetic view generation")
    parser.add_argument("--num-views", type=int, default=8,
                       help="Number of synthetic views to generate")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = Pipeline3D(input_dir=args.input, output_dir=args.output)
    
    # Process images
    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"Error: Image {args.single_image} not found")
            return
        
        pipeline.process_image(
            args.single_image,
            segmentation=args.segmentation,
            depth_model=args.depth_model,
            voxel_resolution=args.resolution,
            apply_texture=not args.no_texture,
            generate_views=not args.no_views,
            num_views=args.num_views
        )
    else:
        pipeline.process_all_images(
            segmentation=args.segmentation,
            depth_model=args.depth_model,
            voxel_resolution=args.resolution,
            apply_texture=not args.no_texture,
            generate_views=not args.no_views,
            num_views=args.num_views
        )


if __name__ == "__main__":
    main()