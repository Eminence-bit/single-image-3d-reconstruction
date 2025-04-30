import os
import torch
import numpy as np
import cv2
import kornia
import matplotlib.pyplot as plt
from pathlib import Path
import time

class ViewSynthesizer:
    """
    Implements SynSin-inspired view synthesis to create multiple views of an object
    using a single image and its depth map.
    """
    
    def __init__(self, device=None):
        """
        Initialize the view synthesizer
        
        Args:
            device: Torch device to use for computations
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"View synthesizer using device: {self.device}")
        
    def preprocess_inputs(self, image, depth, mask=None):
        """
        Preprocess image and depth inputs for view synthesis
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            depth: Depth map as numpy array (H, W)
            mask: Optional mask as numpy array (H, W)
            
        Returns:
            Preprocessed tensors ready for synthesis
        """
        # Convert to torch tensors
        if isinstance(image, np.ndarray):
            # Convert to float and normalize
            image_np = image.astype(np.float32) / 255.0
            # Convert to tensor [B, C, H, W]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
            
        if isinstance(depth, np.ndarray):
            # Normalize depth if needed
            if depth.max() > 1.0:
                depth = depth.astype(np.float32) / 255.0
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            depth_tensor = depth.to(self.device)
            
        # Process mask if provided
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                if mask.max() > 1.0:
                    mask = mask.astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                mask_tensor = mask.to(self.device)
            
        return image_tensor, depth_tensor, mask_tensor
    
    def generate_camera_transformations(self, num_views=8, rotation_range=45, elevation_range=30):
        """
        Generate camera transformations for different views
        
        Args:
            num_views: Number of views to generate
            rotation_range: Maximum rotation angle in degrees
            elevation_range: Maximum elevation angle in degrees
            
        Returns:
            List of camera transformations
        """
        camera_transforms = []
        
        # Generate views in a circle around the object
        for i in range(num_views):
            # Calculate rotation angle (around Y-axis)
            angle_y = np.radians(rotation_range * np.sin(2 * np.pi * i / num_views))
            
            # Calculate elevation angle (around X-axis)
            angle_x = np.radians(elevation_range * np.sin(2 * np.pi * i / num_views))
            
            # For a more interesting pattern, add some rotation around Z-axis too
            angle_z = np.radians(rotation_range/2 * np.sin(np.pi * i / num_views))
            
            # Create rotation matrices
            rot_x = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ], dtype=torch.float32, device=self.device)
            
            rot_y = torch.tensor([
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)]
            ], dtype=torch.float32, device=self.device)
            
            rot_z = torch.tensor([
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device)
            
            # Combine rotations (order matters: Y, X, Z)
            rotation = rot_z @ rot_x @ rot_y
            
            # Translation: shift camera back a bit to keep object in view
            translation = torch.tensor([0, 0, 0.2], dtype=torch.float32, device=self.device)
            
            # Create 4x4 transformation matrix
            transform = torch.eye(4, dtype=torch.float32, device=self.device)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            
            camera_transforms.append(transform)
            
        return camera_transforms
    
    def create_point_cloud(self, image, depth, mask=None, intrinsics=None):
        """
        Create a point cloud from an image and its depth map
        
        Args:
            image: Image tensor [B, C, H, W]
            depth: Depth tensor [B, 1, H, W]
            mask: Optional mask tensor [B, 1, H, W]
            intrinsics: Optional camera intrinsics matrix
            
        Returns:
            points: 3D points [B, 3, N]
            colors: Point colors [B, 3, N]
        """
        batch_size, _, height, width = image.shape
        
        # Create default camera intrinsics if not provided
        # These are approximate values that work for most images
        if intrinsics is None:
            # Focal length is approximated as 1.2 * half the image width
            focal_length = 1.2 * width / 2
            # Principal point at the center of the image
            cx, cy = width / 2, height / 2
            
            intrinsics = torch.tensor([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply mask if provided
        if mask is not None:
            depth = depth * mask
        
        # Use kornia to create a 3D point cloud from the depth map
        # Create meshgrid of pixel coordinates
        xs = torch.linspace(0, width-1, width, device=self.device)
        ys = torch.linspace(0, height-1, height, device=self.device)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')
        
        # Stack coordinates and reshape
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)  # [1, 2, H, W]
        coords = coords.repeat(batch_size, 1, 1, 1)  # [B, 2, H, W]
        
        # Unproject 2D coordinates to 3D space
        # Use kornia's depth_to_3d which handles the computation using camera intrinsics
        points_3d = kornia.geometry.depth.depth_to_3d(depth, intrinsics)  # [B, 3, H, W]
        
        # Reshape points to [B, 3, H*W]
        points = points_3d.reshape(batch_size, 3, -1)
        
        # Extract colors for each point
        colors = image.reshape(batch_size, 3, -1)
        
        # If mask is provided, only keep points inside the mask
        if mask is not None:
            mask_flat = mask.reshape(batch_size, 1, -1)
            mask_idx = (mask_flat > 0.5).repeat(1, 3, 1)
            
            # Extract valid points and colors
            valid_points = []
            valid_colors = []
            
            for b in range(batch_size):
                valid_idx = mask_idx[b, 0]
                valid_points.append(points[b, :, valid_idx])
                valid_colors.append(colors[b, :, valid_idx])
            
            # Stack along batch dimension
            points = torch.stack(valid_points, dim=0)
            colors = torch.stack(valid_colors, dim=0)
        
        return points, colors
    
    def render_new_view(self, points, colors, transform, output_size):
        """
        Render a new view of the point cloud from a specific camera viewpoint
        
        Args:
            points: 3D points [B, 3, N]
            colors: Point colors [B, 3, N]
            transform: Camera transformation matrix [4, 4]
            output_size: Tuple of (height, width) for the output image
            
        Returns:
            Rendered image and depth map
        """
        batch_size = points.shape[0]
        height, width = output_size
        
        # Apply transformation to points
        # Extract rotation and translation from transform
        rotation = transform[:3, :3].unsqueeze(0)
        translation = transform[:3, 3].unsqueeze(0).unsqueeze(-1)
        
        # Apply rotation and translation
        transformed_points = torch.bmm(rotation.repeat(batch_size, 1, 1), points) + translation.repeat(batch_size, 1, 1)
        
        # Project 3D points to 2D image space
        # Simple pinhole camera model
        # Assuming a default focal length of 1.2 * half the image width
        focal_length = 1.2 * width / 2
        
        # Principal point at the center of the image
        cx, cy = width / 2, height / 2
        
        # Project points: [x/z, y/z] * focal_length + [cx, cy]
        z = transformed_points[:, 2:3]  # depths
        x = transformed_points[:, 0:1] / (z + 1e-10) * focal_length + cx
        y = transformed_points[:, 1:2] / (z + 1e-10) * focal_length + cy
        
        # Stack x, y coordinates
        pixel_coords = torch.cat([x, y], dim=1)  # [B, 2, N]
        
        # Initialize output image and depth map
        rendered_image = torch.zeros(batch_size, 3, height, width, device=self.device)
        rendered_depth = torch.ones(batch_size, 1, height, width, device=self.device) * float('inf')
        
        # Create a simple z-buffer renderer
        for b in range(batch_size):
            # Convert pixel coordinates to integers
            coords = pixel_coords[b].t()  # [N, 2]
            px = coords[:, 0].round().long()
            py = coords[:, 1].round().long()
            
            # Filter out points outside the image
            mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px, py = px[mask], py[mask]
            
            # Get z values (depths)
            z_vals = z[b, 0, mask]
            
            # Get colors
            point_colors = colors[b, :, mask]
            
            # For each point, check if it's closer than what's already in the z-buffer
            for i in range(len(px)):
                x_i, y_i = px[i], py[i]
                depth_val = z_vals[i]
                
                # Only update if this point is closer
                if depth_val < rendered_depth[b, 0, y_i, x_i]:
                    rendered_depth[b, 0, y_i, x_i] = depth_val
                    rendered_image[b, :, y_i, x_i] = point_colors[:, i]
        
        # Replace inf values in depth with 0
        rendered_depth[rendered_depth == float('inf')] = 0
        
        # Normalize depth map to [0, 1]
        depth_max = rendered_depth.max()
        if depth_max > 0:
            rendered_depth = rendered_depth / depth_max
        
        return rendered_image, rendered_depth
    
    def generate_synthetic_views(self, image, depth, mask=None, num_views=8, output_size=None):
        """
        Generate multiple synthetic views of an object from different angles
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            depth: Depth map as numpy array (H, W)
            mask: Optional mask as numpy array (H, W)
            num_views: Number of views to generate
            output_size: Optional tuple (height, width) for output size
            
        Returns:
            List of synthetic views (RGB images and depth maps)
        """
        # Preprocess inputs
        image_tensor, depth_tensor, mask_tensor = self.preprocess_inputs(image, depth, mask)
        
        # Set output size if not provided
        if output_size is None:
            _, _, height, width = image_tensor.shape
            output_size = (height, width)
        
        # Create point cloud
        points, colors = self.create_point_cloud(image_tensor, depth_tensor, mask_tensor)
        
        # Generate camera transformations
        camera_transforms = self.generate_camera_transformations(num_views)
        
        # Render synthetic views
        synthetic_views = []
        
        for i, transform in enumerate(camera_transforms):
            # Render new view
            rendered_image, rendered_depth = self.render_new_view(points, colors, transform, output_size)
            
            # Convert to numpy for saving
            rgb_image = rendered_image[0].permute(1, 2, 0).cpu().numpy()
            depth_image = rendered_depth[0, 0].cpu().numpy()
            
            # Clip and convert to uint8 for RGB image
            rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
            
            # Add to results
            synthetic_views.append({
                'rgb': rgb_image,
                'depth': depth_image,
                'angle': i * (360 / num_views)
            })
        
        return synthetic_views
    
    def save_synthetic_views(self, views, output_dir, base_name):
        """
        Save synthetic views to disk
        
        Args:
            views: List of synthetic views (RGB images and depth maps)
            output_dir: Directory to save the views
            base_name: Base name for the output files
            
        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {'rgb': [], 'depth': []}
        
        for i, view in enumerate(views):
            # Save RGB image
            rgb_path = os.path.join(output_dir, f"{base_name}_view_{i:02d}.jpg")
            cv2.imwrite(rgb_path, cv2.cvtColor(view['rgb'], cv2.COLOR_RGB2BGR))
            output_paths['rgb'].append(rgb_path)
            
            # Save depth map
            depth_path = os.path.join(output_dir, f"{base_name}_view_{i:02d}_depth.jpg")
            cv2.imwrite(depth_path, (view['depth'] * 255).astype(np.uint8))
            output_paths['depth'].append(depth_path)
            
        # Create a collage of all views
        self._create_views_collage(views, output_dir, base_name)
        
        return output_paths
    
    def _create_views_collage(self, views, output_dir, base_name):
        """
        Create a collage of all synthetic views
        
        Args:
            views: List of synthetic views
            output_dir: Directory to save the collage
            base_name: Base name for the output file
        """
        num_views = len(views)
        
        # Determine grid size (try to make it square-ish)
        grid_size = int(np.ceil(np.sqrt(num_views)))
        rows, cols = grid_size, grid_size
        
        # Get view dimensions
        view_height, view_width = views[0]['rgb'].shape[:2]
        
        # Create empty canvas
        canvas_rgb = np.zeros((rows * view_height, cols * view_width, 3), dtype=np.uint8)
        canvas_depth = np.zeros((rows * view_height, cols * view_width), dtype=np.uint8)
        
        # Place views in grid
        for i, view in enumerate(views):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            y_start = row * view_height
            y_end = y_start + view_height
            x_start = col * view_width
            x_end = x_start + view_width
            
            # Add RGB image
            canvas_rgb[y_start:y_end, x_start:x_end] = view['rgb']
            
            # Add depth map
            canvas_depth[y_start:y_end, x_start:x_end] = (view['depth'] * 255).astype(np.uint8)
        
        # Save collages
        rgb_collage_path = os.path.join(output_dir, f"{base_name}_views_collage.jpg")
        cv2.imwrite(rgb_collage_path, cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))
        
        depth_collage_path = os.path.join(output_dir, f"{base_name}_depth_collage.jpg")
        cv2.imwrite(depth_collage_path, canvas_depth)
        
        # Create colorized depth collage
        depth_colored = np.zeros_like(canvas_rgb)
        for i in range(rows * view_height):
            for j in range(cols * view_width):
                depth_val = canvas_depth[i, j] / 255.0
                if depth_val > 0:
                    # Apply jet colormap
                    r, g, b = 0, 0, 0
                    if depth_val < 0.25:
                        b = 255
                        g = int(255 * (depth_val / 0.25))
                    elif depth_val < 0.5:
                        b = int(255 * (1 - (depth_val - 0.25) / 0.25))
                        g = 255
                    elif depth_val < 0.75:
                        g = 255
                        r = int(255 * ((depth_val - 0.5) / 0.25))
                    else:
                        g = int(255 * (1 - (depth_val - 0.75) / 0.25))
                        r = 255
                    depth_colored[i, j] = [r, g, b]
        
        depth_color_path = os.path.join(output_dir, f"{base_name}_depth_collage_colored.jpg")
        cv2.imwrite(depth_color_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
        
        print(f"Saved view collages to {rgb_collage_path} and {depth_collage_path}")