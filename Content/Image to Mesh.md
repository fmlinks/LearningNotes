## image to mesh


#### keywords

- Image-to-Mesh Generation
- Direct Mesh Prediction from Medical Images
- Neural Networks for Mesh Generation
- 3D Reconstruction from 2D Medical Images
- End-to-End Medical Image to Mesh Networks
- Deep Learning for Mesh Generation
- MeshCNN 
- Geometric Deep Learning for Medical Imaging
- Point Cloud Generation from Images
- 3D Shape Generation from Medical Images
- Atlas-free Mesh Generation
- Implicit Neural Representations for 3D Objects
- Graph Neural Networks for Mesh Processing
- Occupancy Networks


- [ ] Marching Cubes: for mesh generation
- [ ] implicit

## Related Works

- 3D Rendering (point cloud to image)
- 3D Reconstruction (image to point cloud)
- 3d Mesh Reconstruction


## Simple Benchmark


#### image to mesh
- Pixel2Mesh
- AtlasNet

#### Implicit representation for high quality mesh
- DeepSDF
- Occupancy Networks



#### [MeshCNN](https://ranahanocka.github.io/MeshCNN/)
- mesh to mesh for segmentation or classification;
- dataset on homepage;
- COSEG segmentation dataset;
- Human Segmentation dataset;
- Cubes classification dataset;
- Shrec classification dataset;

#### Pixel2Mesh
- Image-to-mesh, 3D shape reconstruction, Graph convolutional networks (GCNs)

#### AtlasNet
- 2D image to 3D mesh, Image-based 3D reconstruction, Parametric surface modeling

#### DeepSDF, Deep Signed Distance Functions
- image -> Distance Functions -> Marching Cubes -> mesh;
- highlight: smooth, implicit function

#### DISN, Deep Implicit Surface Network
- image -> implicit surface representation -> Marching Cubes -> mesh


#### Occupancy Networks
- image -> Occupancy Functions -> Marching Cubes -> mesh; 
- Implicit surface learning, 3D reconstruction, Occupancy function;
- Generate meshes with arbitrary topology instead of models with fixed topology;

#### Pix2Vox
- output is volume mesh;
- Voxel-based 3D reconstruction, Multi-view 3D reconstruction;


#### Geometric Deep Learning Frameworks (image to mesh tools)
- PyTorch3D;
- Open3D;

#### 3D rendering, Neural 3D Mesh Renderer
- simple mesh -> align with input image (differentiable rendering technique) ->  refined mesh;








## SOTA benchmark




## Potential research direction:

- 3D cardiac image to 4D cardiac mesh
- CT cardiac image to mesh







