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

#### Implicit representation for high-quality mesh
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

### CV

- 2023.6 [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/)
  - Neural surface reconstruction;
  - multi-view stereo algorithms (had been the method of choice for sparse 3D reconstruction.)
  - neural surface reconstruction methods
  - occupancy fields
  - element (smooth) control (simulation BC control): Progressive Levels of Detail. Step size ϵ. As previously discussed, numerical gradients can be interpreted as a smoothing operation where the step size ϵ controls the resolution and the amount of recovered details.

- Instant NGP （Neural GraphicsPrimitives）
- COLMAP
- NeuS
- NeuralWarp

### Med

- 2020 [MedMeshCNN](https://github.com/Divya9Sasidharan/MedMeshCNN)


#### Align raw mesh to target segmentation (seems like the last generation)

- 2020 [voxel2mesh](https://github.com/cvlab-epfl/voxel2mesh)
- 2021.9 [Image-to-Graph Convolutional Network for Deformable Shape Reconstruction from a Single Projection Image](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_25)
- 2023.10 [Abdominal organ segmentation via deep diffeomorphic mesh deformations](https://link.springer.com/article/10.1038/s41598-023-45435-2?fromPaywallRec=false)
- 2024.3 [A Deep Learning Approach for Direct Mesh Reconstruction of Intracranial Arteries](https://studenttheses.uu.nl/handle/20.500.12932/46357)


#### 3D reconstruction

- 2024.4 [Implicit Neural Representations for Breathing-compensated Volume Reconstruction in Robotic Ultrasound](https://arxiv.org/abs/2311.04999)

- 2024.9 [Neural implicit surface reconstruction of freehand 3D ultrasound volume with geometric constraints](https://www.sciencedirect.com/science/article/pii/S1361841524002305?casa_token=NhMAc1QEx4IAAAAA:Q9ovrbow9Jo2wbvHR6i-whY3Va9_anbWN9pGUGqcdyUOHQTSHl3-m6SGEPhg65Kvt7wlOVjgHVY)
  - signed distance functions (SDFs) 
  - implicit neural representation (INR)
  - Neural volume reconstruction. 
  - Neural surface reconstruction















## Dataset - 3D Reconstruction - CV

[DTU](https://paperswithcode.com/sota/3d-reconstruction-on-dtu)




## Potential research direction:

- 3D cardiac image to 4D cardiac mesh
- CT cardiac image to mesh







