import os
import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path

class VoxelToMesh:
    def __init__(self, voxels_dir="data/output/voxels", output_dir="data/output/meshes"):
        """
        Initialize the voxel to mesh converter
        
        Args:
            voxels_dir: Directory containing voxel grid .npy files
            output_dir: Directory to save output mesh files
        """
        self.voxels_dir = voxels_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def apply_marching_cubes(self, voxel_grid, threshold=0.5, step_size=1):
        """
        Apply the Marching Cubes algorithm to convert a voxel grid to a mesh
        
        Args:
            voxel_grid: 3D numpy array representing voxel grid
            threshold: Isosurface threshold value
            step_size: Step size for the algorithm (can be used to reduce mesh complexity)
            
        Returns:
            vertices: Mesh vertices
            faces: Mesh faces
            normals: Surface normals at each vertex
        """
        # Pad the voxel grid to avoid boundary issues
        padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
        
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
    
    def convert_and_save(self, voxel_path, output_base_name=None):
        """
        Convert a voxel grid to a mesh and save it in various formats
        
        Args:
            voxel_path: Path to the voxel grid .npy file
            output_base_name: Base name for output files (if None, use input file name)
            
        Returns:
            Dictionary with paths to saved mesh files
        """
        print(f"Converting voxel grid from {voxel_path} to mesh...")
        
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
            output_base_name = Path(voxel_path).stem.replace("_voxels", "")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        # Post-process the mesh
        print("Applying mesh post-processing...")
        processed_mesh = self.post_process_mesh(mesh)
        
        # Save in multiple formats (both original and processed)
        output_files = {}
        
        # Save original mesh
        obj_path = os.path.join(self.output_dir, f"{output_base_name}.obj")
        mesh.export(obj_path)
        output_files['obj_original'] = obj_path
        print(f"Saved original mesh as OBJ: {obj_path}")
        
        # Save processed mesh in multiple formats
        obj_path_processed = os.path.join(self.output_dir, f"{output_base_name}_processed.obj")
        processed_mesh.export(obj_path_processed)
        output_files['obj_processed'] = obj_path_processed
        print(f"Saved processed mesh as OBJ: {obj_path_processed}")
        
        stl_path_processed = os.path.join(self.output_dir, f"{output_base_name}_processed.stl")
        processed_mesh.export(stl_path_processed)
        output_files['stl_processed'] = stl_path_processed
        print(f"Saved processed mesh as STL: {stl_path_processed}")
        
        ply_path_processed = os.path.join(self.output_dir, f"{output_base_name}_processed.ply")
        processed_mesh.export(ply_path_processed)
        output_files['ply_processed'] = ply_path_processed
        print(f"Saved processed mesh as PLY: {ply_path_processed}")
        
        # Generate comparison previews
        preview_path_original = os.path.join(self.output_dir, f"{output_base_name}_preview.png")
        self.create_mesh_preview(mesh, preview_path_original)
        output_files['preview_original'] = preview_path_original
        
        preview_path_processed = os.path.join(self.output_dir, f"{output_base_name}_processed_preview.png")
        self.create_mesh_preview(processed_mesh, preview_path_processed)
        output_files['preview_processed'] = preview_path_processed
        
        return output_files
    
    def post_process_mesh(self, mesh, smooth_iterations=5, target_faces=None, fill_hole_size=100):
        """
        Apply post-processing operations to improve mesh quality
        
        Args:
            mesh: Trimesh object
            smooth_iterations: Number of Laplacian smoothing iterations
            target_faces: Target number of faces for decimation (if None, use 1/3 of original)
            fill_hole_size: Maximum hole size to fill
        
        Returns:
            Processed Trimesh object
        """
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        processed_mesh = mesh.copy()
        
        # Fill holes first (easier to fill before decimation)
        try:
            # Make sure mesh is watertight
            if not processed_mesh.is_watertight:
                print(f"Filling holes (max size: {fill_hole_size})...")
                processed_mesh.fill_holes()
                print(f"After hole filling: {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces")
        except Exception as e:
            print(f"Error during hole filling: {e}")
        
        # Apply smoothing to reduce jagged edges
        try:
            print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
            processed_mesh = processed_mesh.smoothed(method='laplacian', iterations=smooth_iterations)
            print(f"After smoothing: {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces")
        except Exception as e:
            print(f"Error during smoothing: {e}")
        
        # Apply decimation to reduce number of triangles
        try:
            if target_faces is None:
                # Default to 1/3 of original face count, but not less than 5000
                target_faces = max(5000, len(processed_mesh.faces) // 3)
            
            if len(processed_mesh.faces) > target_faces:
                print(f"Applying decimation (target: {target_faces} faces)...")
                processed_mesh = processed_mesh.simplify_quadric_decimation(target_faces)
                print(f"After decimation: {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces")
        except Exception as e:
            print(f"Error during decimation: {e}")
        
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
            if file.endswith("_voxels.npy"):
                voxel_path = os.path.join(self.voxels_dir, file)
                output_files = self.convert_and_save(voxel_path)
                if output_files:
                    processed_files.append(output_files)
        
        print(f"Successfully processed {len(processed_files)} voxel grid files")
        return processed_files

if __name__ == "__main__":
    # Create converter
    converter = VoxelToMesh(
        voxels_dir="data/output/voxels",
        output_dir="data/output/meshes"
    )
    
    # Process all voxel grid files
    converter.process_all_voxel_files()