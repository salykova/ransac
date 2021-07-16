# Plane fitting with RANSAC (Random Sample Consensus) algorithm
The goal of this project is to find the dominant plane (i.e. the floor) in the given pointclouds, as well as extracting multiple planes from more complex scenes.

### Dominant plane

<img src="https://user-images.githubusercontent.com/63703454/125992080-6f0fd452-5428-48c5-b114-75870cbd12ed.png" width="400" height="250"> <img src="https://user-images.githubusercontent.com/63703454/125992626-f844e0cc-2930-434f-a768-289dd7407893.png" width="400" height="250">

Example pointcloud from the dataset

<img src="https://user-images.githubusercontent.com/63703454/125992243-e61f96f9-108b-4872-a3ce-fc3392e58faa.png" width="500" height="300">
Succesfully detected plane and colored the inliers with red of the down-sampled point cloud.

### Multiple planes

<img src="https://user-images.githubusercontent.com/63703454/125992959-ad8f24cc-df16-4d92-b09c-315a8a3a8628.png" width="500" height="400">
<img src="https://user-images.githubusercontent.com/63703454/125992980-db76c173-2ed5-4789-a2c0-163ee7ea8b38.png" width="500" height="400">

Example pointcloud with multiple planes visible. Example result on down-sampled point cloud with planes in different colors.
