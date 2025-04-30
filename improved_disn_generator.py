import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch

class DISN3DGenerator:
    def __init__(self, resolution=128):  # Increased resolution for better detail
        """
        Initialize the DISN 3D voxel grid generator
        
        Args:
            resolution: The resolution of the voxel grid (increased for better detail)
        """
        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def create_voxel_grid_from_depth(self, depth_map, original_image=None, mask=None, depth_scale=1.0):
        """
        Convert a depth map to a 3D voxel grid using DISN-inspired approach with improved shape preservation
        
        Args:
            depth_map: A depth map as a numpy array (HxW)
            original_image: The original RGB image (for shape reference)
            mask: Optional binary mask to isolate object (HxW)
            depth_scale: Scaling factor for depth values
            
        Returns:
            voxel_grid: A numpy array (resolution x resolution x resolution)
        """
        # If no mask is provided, create a simple one (all foreground)
        if mask is None:
            mask = np.ones_like(depth_map)
            
        # If original image is provided, use it for better shape preservation
        if original_image is not None:
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                # Convert to grayscale for processing
                img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                # Use image edges to enhance shape definition
                edges = cv2.Canny(img_gray, 100, 200)
                # Combine edge information with mask
                mask = np.maximum(mask, edges / 255.0)
        
        # Normalize depth map to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / 255.0
            
        # Apply the mask
        masked_depth = depth_map * (mask > 0)
        
        # Remove noise and small isolated areas in the depth map
        kernel = np.ones((3,3), np.uint8)
        cleaned_depth = cv2.morphologyEx(masked_depth.astype(np.float32), cv2.MORPH_OPEN, kernel)
        
        # Convert to torch tensor
        depth_tensor = torch.from_numpy(cleaned_depth).float().to(self.device)
        
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
        
        # Use a more sophisticated approach to create the 3D volume
        # Instead of just filling from front to back, create a solid object based on the depth profile
        for x in range(self.resolution):
            for y in range(self.resolution):
                # Get the depth at this pixel
                depth_val = depth_resized[y, x].item() * depth_scale
                
                if depth_val > 0.01:  # Only process non-background pixels
                    # Convert to voxel coordinate in Z dimension
                    z_val = int(depth_val * self.resolution)
                    
                    # Create a solid object by filling a range around the depth value
                    # This creates more of a solid object than just a shell
                    z_range = max(2, int(self.resolution * 0.05))  # Minimum thickness
                    
                    z_start = max(0, z_val - z_range)
                    z_end = min(self.resolution - 1, z_val + z_range)
                    
                    # Create a gradient falloff for more natural shape
                    falloff = np.ones(z_end - z_start + 1)
                    
                    # Apply the values to the voxel grid
                    voxel_grid[y, x, z_start:z_end+1] = torch.from_numpy(falloff).to(self.device)
        
        # Apply 3D smoothing to create a more natural shape
        voxel_grid = self.smooth_voxel_grid(voxel_grid)
        
        return voxel_grid.cpu().numpy()
    
    def smooth_voxel_grid(self, voxel_grid, kernel_size=3):
        """
        Apply 3D smoothing to the voxel grid for more natural shapes
        
        Args:
            voxel_grid: The voxel grid tensor
            kernel_size: Size of the smoothing kernel
            
        Returns:
            Smoothed voxel grid
        """
        # Create a simple 3D averaging filter
        padding = kernel_size // 2
        
        # Use 3D convolution as a smoothing operation
        smoothing_filter = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), 
                                    device=self.device) / (kernel_size ** 3)
        
        # Add batch and channel dimensions
        grid_5d = voxel_grid.unsqueeze(0).unsqueeze(0)
        
        # Apply 3D convolution for smoothing
        smoothed_grid = torch.nn.functional.conv3d(
            grid_5d, 
            smoothing_filter, 
            padding=padding
        )
        
        # Remove batch and channel dimensions
        return smoothed_grid.squeeze()
    
    def generate_3d_voxels_from_depth_file(self, depth_path, original_image_path=None, mask_path=None, output_dir="data/output/voxels"):
        """
        Generate a 3D voxel grid from a depth map file with reference to original image
        
        Args:
            depth_path: Path to the depth map image
            original_image_path: Path to the original image (for shape reference)
            mask_path: Optional path to a mask image
            output_dir: Directory to save the output
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load depth map
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Load original image if provided
        original_image = None
        if original_image_path and os.path.exists(original_image_path):
            original_image = cv2.imread(original_image_path)
            print(f"Using original image for reference: {original_image_path}")
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Using mask: {mask_path}")
        
        # Generate voxel grid
        print(f"Generating 3D voxel grid for {depth_path}...")
        start_time = time.time()
        voxel_grid = self.create_voxel_grid_from_depth(depth_map, original_image, mask)
        end_time = time.time()
        print(f"Voxel grid generation completed in {end_time - start_time:.2f} seconds")
        
        # Save the voxel grid as a NumPy file
        output_name = Path(depth_path).stem.replace("_depth", "")
        output_path = f"{output_dir}/{output_name}_improved_voxels.npy"
        np.save(output_path, voxel_grid)
        print(f"Saved improved voxel grid to {output_path}")
        
        # Visualize the voxel grid (3 slices: XY, YZ, XZ)
        self.visualize_voxel_grid(voxel_grid, output_dir, f"{output_name}_improved")
        
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
        axes[0].imshow(voxel_grid[mid_x, :, :], cmap='jet')
        axes[0].set_title(f'YZ Slice (X={mid_x})')
        
        axes[1].imshow(voxel_grid[:, mid_y, :], cmap='jet')
        axes[1].set_title(f'XZ Slice (Y={mid_y})')
        
        axes[2].imshow(voxel_grid[:, :, mid_z], cmap='jet')
        axes[2].set_title(f'XY Slice (Z={mid_z})')
        
        # Save the figure
        viz_path = f"{output_dir}/{output_name}_voxel_slices.png"
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved visualization to {viz_path}")
    
    def process_depth_maps(self, input_dir="data/output/depth/large", pattern="_depth.jpg", 
                          output_dir="data/output/voxels_improved"):
        """
        Process all depth maps in the input directory with improved shape preservation
        
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
                
                # Find matching original image and mask
                base_name = img_file.replace('_depth.jpg', '')
                
                # Look for original image with different potential naming patterns
                original_paths = [
                    os.path.join("data/output", f"{base_name}.jpg"),
                    os.path.join("data/output", f"{base_name}_segmented.jpg"),
                    os.path.join("data/input", f"{base_name.replace('_masks', '')}.jpg"),
                    os.path.join("data/input", f"{base_name.replace('_masks', '')}.png")
                ]
                
                original_image_path = next((p for p in original_paths if os.path.exists(p)), None)
                
                # Find a matching mask
                mask_paths = [
                    os.path.join("data/output", f"{base_name}.jpg"),
                    os.path.join("data/output", f"{base_name}_masks.jpg")
                ]
                
                mask_path = next((p for p in mask_paths if os.path.exists(p)), None)
                
                output_path = self.generate_3d_voxels_from_depth_file(
                    depth_path, 
                    original_image_path, 
                    mask_path, 
                    output_dir
                )
                processed_files.append(output_path)
        
        return processed_files

if __name__ == "__main__":
    # Create DISN 3D voxel grid generator with higher resolution
    generator = DISN3DGenerator(resolution=128)  # Increased resolution for better detail
    
    # Process depth maps and generate improved 3D voxel grids
    output_dir = "data/output/voxels_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all depth maps in the large directory (DPT_Large model)
    generator.process_depth_maps(input_dir="data/output/depth/large", 
                               pattern="_depth.jpg",
                               output_dir=output_dir)