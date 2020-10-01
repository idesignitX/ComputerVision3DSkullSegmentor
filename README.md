# ComputerVision3DSkullSegmentor
A python program that segments the skull from slice images of a human head and generates a 3D model of the skull

## Introduction
In this work we have created a module that performs 3D reconstruction of the human brain. We initially
perform image segmentation MRI scans of the brain and then generated a 3D model by compositing the
segmentation results into a point cloud and representing it in an interactive 3D interface. This work acts as
a proof of concept for 3-D reconstruction and can be used as a model for any 2-D sliced data. For example,
we can use this moel to reconstruct any real-life organs provided we get their sliced images.
Such reconstruction can help to visualize the organ more intuitively and thus in case of any medical issues
with the organs, helps in better diagnosis.

## High Level Details
* On a high level, the project involves two modules, the segmentation module and the reconstruction module.
* The segmentation module implements the Snake segmentation module in python and generates contours for each sliced image.
* The reconstruction module uses Open GL to stitch the points between contours of successive layers and generate the reconstructed image. Here, we have used the triangulation method, i.e. generate a triangle between two points of one layer and one on the next layer.

## Configurations and controllable parameters
* We have made the implementation as generic as possible by providing a configuration json that
controls the behavior of the codes.
* It contains attributes like alpha, beta, gamma, max iterations, folder location of the images, the yseparation (horizontal distance) between layers and a unique identifier for each run called session
name. More details about the configuration json can be found in the README file.
* The 3D reconstruction that we have performed provides an interactive interface via pygame. We
have provided controls to change color, orientation, display/hide lines/triangles/solid lines formed,
expand/contract the image etc. The README file provides more details about the possible controls.
The images shown below in results section shows images

## Results
Note that in the pictures below, the segmented 3D slices have been given extra separation for detail, hence the shape is elongated. This is configurable in the program.

![Segmentation of a 2D slice](https://github.com/richan8/ComputerVision3DSkullSegmentor/blob/main/imgs/1.PNG)
![3D Reconstruction](https://github.com/richan8/ComputerVision3DSkullSegmentor/blob/main/imgs/2.PNG)
