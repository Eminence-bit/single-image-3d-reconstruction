# Single-Image 3D Reconstruction

A comprehensive pipeline for transforming 2D images into textured 3D models using AI-powered computer vision techniques.

## Project Overview

This project implements an end-to-end pipeline for reconstructing 3D models from single 2D images. The system utilizes multiple state-of-the-art computer vision techniques including object segmentation, depth estimation, synthetic view generation, voxel grid creation, and texture mapping to create detailed 3D representations from ordinary 2D photographs.

### Project Motivation

Traditional 3D reconstruction often requires multiple images or specialized equipment. This project aims to overcome these limitations by leveraging recent advances in AI and computer vision to generate plausible 3D models from a single image, making 3D content creation more accessible.

## Technical Architecture

### Pipeline Components and Workflow

1. **Object Segmentation (`segmentation.py`)**

   - Implements Mask R-CNN to isolate objects from backgrounds
   - Pre-trained on COCO dataset for recognizing 80+ common object categories
   - Outputs binary masks and segmented images

2. **Depth Estimation (`depth_estimation.py`, `depth_generator.py`)**

   - Utilizes MiDaS DPT_Large model for monocular depth estimation
   - Transforms 2D pixel information into relative depth values
   - Generates both raw depth maps and colored visualizations

3. **View Synthesis (`view_synthesizer.py`)**

   - Creates 8 synthetic viewpoints around the object
   - Implements novel view synthesis using 3D point cloud reprojection
   - Handles occlusion and performs hole-filling to create coherent views

4. **Voxel Grid Creation (`voxel_mesh.py`)**

   - Integrates multiple synthetic views into a consistent 3D representation
   - Implements space carving and volumetric fusion techniques
   - Creates 128³ resolution voxel grids with occupancy probability values

5. **Mesh Generation (`improved_voxel_to_mesh.py`)**

   - Converts voxel grids to triangular meshes using Marching Cubes algorithm
   - Applies Laplacian smoothing to reduce voxelization artifacts
   - Implements spherical UV mapping for texture projection

6. **Texture Mapping**
   - Projects original image colors onto the 3D mesh
   - Enhances color vibrancy and detail preservation
   - Generates textured meshes in standard 3D formats (OBJ and PLY)

### Algorithm Details

#### Object Segmentation

The segmentation module uses Mask R-CNN with a ResNet50-FPN backbone. The model performs instance segmentation, providing pixel-precise masks for detected objects. The system automatically selects the most prominent object based on mask area and confidence score.

#### Depth Estimation

The depth estimation pipeline uses MiDaS DPT_Large, a transformer-based architecture trained on diverse datasets. While the raw output provides relative depth values, the system applies normalization and refinement to enhance depth consistency at object boundaries.

#### Multi-View Synthesis

The view synthesis module implements a geometric approach where:

1. The original image and depth map are converted to a 3D point cloud
2. The point cloud is rotated around the vertical axis at predefined angles
3. Points are reprojected onto new virtual camera planes
4. Missing regions are filled using nearest-neighbor interpolation

#### Voxel Grid Construction

The voxel grid creation follows a probabilistic space carving approach:

1. Initialize an empty 128³ voxel grid
2. For each synthetic view:
   - Project rays from the camera through each pixel
   - Update voxel occupancy probabilities based on depth values
3. Combine evidence from all views using a weighted average
4. Apply threshold to determine final occupancy

#### Mesh Generation and Texturing

The mesh generation process:

1. Applies Marching Cubes algorithm to extract isosurfaces from the voxel grid
2. Performs Laplacian mesh smoothing to reduce staircase artifacts
3. Creates UV coordinates using spherical projection
4. Maps the original image texture onto the mesh using these coordinates
5. Exports in standard 3D formats (OBJ/PLY) with material files

### Key Technical Challenges and Solutions

#### Handling Occlusions

- Challenge: Synthesizing views reveals previously occluded regions
- Solution: Implements hole-filling algorithms and geometric consistency checks

#### Voxel Resolution vs. Performance

- Challenge: Higher voxel resolutions create more detailed models but require more computation
- Solution: Uses an adaptive approach with 128³ resolution as a balanced default

#### Texture Mapping Discontinuities

- Challenge: Projecting 2D textures onto complex 3D geometries can cause seams
- Solution: Implements spherical UV mapping with blending at boundaries

## Features

- **Object Segmentation**: Isolates objects using Mask R-CNN
- **Depth Estimation**: Generates depth maps using MiDaS DPT_Large
- **Multi-View Synthesis**: Creates multiple views from a single image
- **Volumetric Reconstruction**: Builds voxel grids from synthesized views
- **3D Mesh Generation**: Creates textured 3D meshes in OBJ and PLY formats
- **Batch Processing**: Automatically processes all images in the input directory

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended but not required)

## Installation

1. Clone this repository

```bash
git clone https://github.com/eminence-bit/single-image-3d-reconstruction.git
cd single-image-3d-reconstruction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download pre-trained models (if not included)
   - SAM ViT-B model (sam_vit_b_01ec64.pth) should be placed in the root directory
   - YOLOv8n weights (yolov8n.pt) should be placed in the root directory

## Usage

1. Place your input images in the `data/input/` directory

2. Run the main script:

```bash
python main.py
```

3. The results will be saved in the `data/output/` directory with the following structure:
   - `data/output/` - Segmentation results
   - `data/output/depth/` - Depth maps
   - `data/output/views/` - Synthetic views
   - `data/output/voxels/` - Voxel visualizations
   - `data/output/meshes/` - 3D meshes (OBJ and PLY formats)

## Configuration

Advanced settings can be modified in `main.py`:

- Image resolution
- Number of synthetic views
- Voxel grid resolution
- Mesh smoothing parameters
- Texture mapping options

## Experimental Results

The system was evaluated on a diverse set of test images with the following results:

- **Reconstruction Quality**: The system produces recognizable 3D shapes for most common objects, with best results for objects with convex geometries.
- **Processing Time**: On a typical CPU system, processing one image takes approximately 2-5 minutes depending on resolution.
- **Model Size**: Generated meshes typically contain 5,000-15,000 triangles, making them suitable for web and mobile applications.

### Comparison to Similar Systems

Compared to other single-image 3D reconstruction systems, this implementation:

- Requires no specialized training on 3D datasets
- Works on arbitrary object categories
- Produces textured meshes rather than just depth or point clouds
- Uses an efficient pipeline suitable for consumer hardware

## Example Workflow and Results

The pipeline processes images through the following stages:

1. **Input Image**: A standard 2D photograph
2. **Segmentation**: Object isolation from background
3. **Depth Map**: Grayscale representation of depth values
4. **Synthetic Views**: Multiple perspectives generated from single image
5. **Voxel Grid**: 3D volumetric representation
6. **Textured Mesh**: Final 3D model with original image colors applied

Check the output directories to see the results from your processed images.

## Limitations and Future Work

### Current Limitations

- Works best with well-isolated objects
- Performance depends on image quality and clarity
- Complex shapes with thin structures may not reconstruct perfectly
- Symmetry assumptions can lead to inaccuracies in unseen parts

### Future Improvements

- Integration of learned priors for better reconstruction of occluded parts
- Implementation of neural rendering techniques for higher-quality texturing
- Support for multi-view input to enhance accuracy
- Optimization for real-time performance on mobile devices

## Troubleshooting

If you encounter the error "No module named 'pyglet'", this only affects the interactive 3D viewer and doesn't impact the generation of 3D models. The system will still save mesh previews using matplotlib.

## Acknowledgments

- MiDaS depth estimation model by Intel ISL
- Mask R-CNN implementation from Torchvision
- YOLO object detection from Ultralytics
- Segment Anything Model (SAM) from Meta AI Research

## Project Contributors

This project was developed as a minor project by:

- Name: G Prajyoth
- Course: Artificial Intelligence and Machine Learning
- Institution: VBIT
- Year: 2025
