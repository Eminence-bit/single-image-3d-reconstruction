import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time

class DISN3DGenerator:
    def __init__(self, resolution=64):
        """
        Initialize the DISN 3D voxel grid generator
        
        Args:
            resolution: The resolution of the voxel grid (e.g., 64 means 64x64x64 voxels)
        """
        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def create_voxel_grid_from_depth(self, depth_map, mask=None, depth_scale=1.0):
        """
        Convert a depth map to a 3D voxel grid using DISN-inspired approach
        
        Args:
            depth_map: A depth map as a numpy array (HxW)
            mask: Optional binary mask to isolate object (HxW)
            depth_scale: Scaling factor for depth values
            
        Returns:
            voxel_grid: A numpy array (resolution x resolution x resolution)
        """
        # If no mask is provided, create a simple one (all foreground)
        if mask is None:
            mask = np.ones_like(depth_map)
        
        # Normalize depth map to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / 255.0
            
        # Apply the mask
        masked_depth = depth_map * (mask > 0)
        
        # Convert to torch tensor
        depth_tensor = torch.from_numpy(masked_depth).float().to(self.device)
        
        # Resize to match our voxel resolution for the XY dimensions
        depth_resized = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Create a 3D grid using the depth map
        voxel_grid = torch.zeros((self.resolution, self.resolution, self.resolution), 
                                device=self.device)
        
        # For each pixel in the depth map, fill voxels up to the depth value
        for x in range(self.resolution):
            for y in range(self.resolution):
                # Get the depth at this pixel
                depth_val = depth_resized[y, x].item() * depth_scale
                # Convert to voxel coordinate in Z dimension
                z_end = min(int(depth_val * self.resolution), self.resolution-1)
                # Fill voxels from the front to the depth value
                voxel_grid[y, x, :z_end+1] = 1.0
        
        return voxel_grid.cpu().numpy()
    
    def generate_3d_voxels_from_depth_file(self, depth_path, mask_path=None, output_dir="data/output/voxels"):
        """
        Generate a 3D voxel grid from a depth map file
        
        Args:
            depth_path: Path to the depth map image
            mask_path: Optional path to a mask image
            output_dir: Directory to save the output
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load depth map
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Generate voxel grid
        print(f"Generating 3D voxel grid for {depth_path}...")
        start_time = time.time()
        voxel_grid = self.create_voxel_grid_from_depth(depth_map, mask)
        end_time = time.time()
        print(f"Voxel grid generation completed in {end_time - start_time:.2f} seconds")
        
        # Save the voxel grid as a NumPy file
        output_name = Path(depth_path).stem.replace("_depth", "")
        output_path = f"{output_dir}/{output_name}_voxels.npy"
        np.save(output_path, voxel_grid)
        print(f"Saved voxel grid to {output_path}")
        
        # Visualize the voxel grid (3 slices: XY, YZ, XZ)
        self.visualize_voxel_grid(voxel_grid, output_dir, output_name)
        
        return output_path
    
    def visualize_voxel_grid(self, voxel_grid, output_dir, output_name):
        """
        Visualize the voxel grid by creating slices through it
        
        Args:
            voxel_grid: The 3D voxel grid as a numpy array
            output_dir: Directory to save the visualizations
            output_name: Base name for the output files
        """
        # Create a figure with 3 subplots for the 3 orthogonal slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Take central slices from each dimension
        mid_x = voxel_grid.shape[0] // 2
        mid_y = voxel_grid.shape[1] // 2
        mid_z = voxel_grid.shape[2] // 2
        
        # Create visualizations for each slice
        axes[0].imshow(voxel_grid[mid_x, :, :], cmap='gray')
        axes[0].set_title(f'YZ Slice (X={mid_x})')
        
        axes[1].imshow(voxel_grid[:, mid_y, :], cmap='gray')
        axes[1].set_title(f'XZ Slice (Y={mid_y})')
        
        axes[2].imshow(voxel_grid[:, :, mid_z], cmap='gray')
        axes[2].set_title(f'XY Slice (Z={mid_z})')
        
        # Save the figure
        viz_path = f"{output_dir}/{output_name}_voxel_slices.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved visualization to {viz_path}")
    
    def process_depth_maps(self, input_dir="data/output/depth/large", pattern="_depth.jpg", 
                          output_dir="data/output/voxels"):
        """
        Process all depth maps in the input directory
        
        Args:
            input_dir: Directory containing depth maps
            pattern: Pattern to match depth map files
            output_dir: Directory to save the output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        processed_files = []
        
        for img_file in os.listdir(input_dir):
            if pattern in img_file and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                depth_path = os.path.join(input_dir, img_file)
                
                # Try to find a matching mask file
                base_name = img_file.replace('_depth.jpg', '')
                mask_path = os.path.join("data/output", f"{base_name}.jpg")
                if not os.path.exists(mask_path):
                    mask_path = None
                
                output_path = self.generate_3d_voxels_from_depth_file(depth_path, mask_path, output_dir)
                processed_files.append(output_path)
        
        return processed_files
    
    def export_binary_volume(self, voxel_grid, output_path):
        """
        Export the voxel grid as a binary volume file
        
        Args:
            voxel_grid: The 3D voxel grid as a numpy array
            output_path: Path to save the binary volume
        """
        # Convert to binary volume (1 byte per voxel)
        binary_volume = (voxel_grid > 0.5).astype(np.uint8)
        
        # Save as a raw binary file
        with open(output_path, 'wb') as f:
            f.write(binary_volume.tobytes())
        
        # Also save metadata file
        meta_path = output_path + '.json'
        import json
        with open(meta_path, 'w') as f:
            json.dump({
                'dimensions': voxel_grid.shape,
                'voxel_size': 1.0,
                'format': 'binary',
                'data_type': 'uint8'
            }, f)
        
        print(f"Exported binary volume to {output_path}")
        print(f"Exported metadata to {meta_path}")
        
        return output_path

if __name__ == "__main__":
    # Create DISN 3D voxel grid generator
    generator = DISN3DGenerator(resolution=64)  # 64x64x64 voxel grid
    
    # Process depth maps and generate 3D voxel grids
    output_dir = "data/output/voxels"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all depth maps in the large directory (DPT_Large model)
    generator.process_depth_maps(input_dir="data/output/depth/large", 
                               pattern="_depth.jpg",
                               output_dir=output_dir)