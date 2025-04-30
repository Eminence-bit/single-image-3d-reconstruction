import os
import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path

class ImprovedVoxelToMesh:
    def __init__(self, voxels_dir="data/output/voxels_improved", output_dir="data/output/meshes_improved"):
        """
        Initialize the improved voxel to mesh converter
        
        Args:
            voxels_dir: Directory containing improved voxel grid .npy files
            output_dir: Directory to save output mesh files
        """
        self.voxels_dir = voxels_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def apply_marching_cubes(self, voxel_grid, threshold=0.3, step_size=1):
        """
        Apply the Marching Cubes algorithm to convert a voxel grid to a mesh
        
        Args:
            voxel_grid: 3D numpy array representing voxel grid
            threshold: Isosurface threshold value (lowered to capture more detail)
            step_size: Step size for the algorithm (can be used to reduce mesh complexity)
            
        Returns:
            vertices: Mesh vertices
            faces: Mesh faces
            normals: Surface normals at each vertex
        """
        # Smooth the voxel grid slightly to reduce artifacts
        from scipy import ndimage
        smoothed_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.8)
        
        # Pad the voxel grid to avoid boundary issues
        padded_grid = np.pad(smoothed_grid, 1, mode='constant', constant_values=0)
        
        try:
            # Apply marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(
                padded_grid, 
                level=threshold, 
                step_size=step_size,
                allow_degenerate=False
            )
            
            # Adjust vertices to account for padding
            vertices = vertices - 1
            
            return vertices, faces, normals
        except Exception as e:
            print(f"Error applying marching cubes: {e}")
            return None, None, None
    
    def post_process_mesh(self, mesh, smooth_iterations=3):
        """
        Apply post-processing operations to improve mesh quality
        
        Args:
            mesh: Trimesh object
            smooth_iterations: Number of Laplacian smoothing iterations
            
        Returns:
            Processed Trimesh object
        """
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        processed_mesh = mesh.copy()
        
        # Apply smoothing to reduce jagged edges
        try:
            print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
            processed_mesh = processed_mesh.smoothed(method='laplacian', iterations=smooth_iterations)
            print(f"After smoothing: {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces")
        except Exception as e:
            print(f"Error during smoothing: {e}")
        
        # Final cleanup and normals recalculation
        try:
            processed_mesh.remove_degenerate_faces()
            processed_mesh.remove_duplicate_faces()
            processed_mesh.remove_unreferenced_vertices()
            processed_mesh.fix_normals()
        except Exception as e:
            print(f"Error during final cleanup: {e}")
        
        print(f"Final processed mesh: {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces")
        return processed_mesh
    
    def convert_and_save(self, voxel_path, output_base_name=None):
        """
        Convert a voxel grid to a mesh and save it in various formats
        
        Args:
            voxel_path: Path to the voxel grid .npy file
            output_base_name: Base name for output files (if None, use input file name)
            
        Returns:
            Dictionary with paths to saved mesh files
        """
        print(f"Converting improved voxel grid from {voxel_path} to mesh...")
        
        # Load the voxel grid
        try:
            voxel_grid = np.load(voxel_path)
        except Exception as e:
            print(f"Error loading voxel grid: {e}")
            return None
        
        print(f"Loaded voxel grid with shape {voxel_grid.shape}")
        
        # Apply marching cubes
        vertices, faces, normals = self.apply_marching_cubes(voxel_grid)
        
        if vertices is None or faces is None:
            print("Failed to generate mesh. Skipping.")
            return None
        
        print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
        
        # Set output base name
        if output_base_name is None:
            output_base_name = Path(voxel_path).stem.replace("_improved_voxels", "")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        # Post-process the mesh
        print("Applying mesh post-processing...")
        processed_mesh = self.post_process_mesh(mesh)
        
        # Save processed mesh in multiple formats
        output_files = {}
        
        obj_path = os.path.join(self.output_dir, f"{output_base_name}_improved.obj")
        processed_mesh.export(obj_path)
        output_files['obj'] = obj_path
        print(f"Saved improved mesh as OBJ: {obj_path}")
        
        stl_path = os.path.join(self.output_dir, f"{output_base_name}_improved.stl")
        processed_mesh.export(stl_path)
        output_files['stl'] = stl_path
        print(f"Saved improved mesh as STL: {stl_path}")
        
        # Generate preview
        preview_path = os.path.join(self.output_dir, f"{output_base_name}_improved_preview.png")
        self.create_mesh_preview(processed_mesh, preview_path)
        output_files['preview'] = preview_path
        
        return output_files
    
    def create_mesh_preview(self, mesh, output_path):
        """
        Create a preview image of the mesh
        
        Args:
            mesh: Trimesh object
            output_path: Path to save the preview image
        """
        # Create a scene with the mesh
        scene = trimesh.Scene(mesh)
        
        # Save a simple rendering
        try:
            # Try to render with trimesh's built-in renderer
            png = scene.save_image(resolution=[1024, 768], visible=True)
            with open(output_path, 'wb') as f:
                f.write(png)
            print(f"Saved mesh preview: {output_path}")
        except Exception as e:
            print(f"Error creating mesh preview: {e}")
            # If that fails, try with matplotlib
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
                print(f"Failed to create mesh preview with matplotlib: {e2}")
    
    def process_all_voxel_files(self):
        """
        Process all voxel grid files in the voxels directory
        """
        processed_files = []
        
        for file in os.listdir(self.voxels_dir):
            if file.endswith("_improved_voxels.npy"):
                voxel_path = os.path.join(self.voxels_dir, file)
                output_files = self.convert_and_save(voxel_path)
                if output_files:
                    processed_files.append(output_files)
        
        print(f"Successfully processed {len(processed_files)} voxel grid files")
        return processed_files

if __name__ == "__main__":
    # Create improved converter
    converter = ImprovedVoxelToMesh(
        voxels_dir="data/output/voxels_improved",
        output_dir="data/output/meshes_improved"
    )
    
    # Process all improved voxel grid files
    converter.process_all_voxel_files()