# TorchGeo in ArcGIS Pro

## Project Description

The aim of this project is to implement TorchGeo in ArcGIS Pro in the form of a toolbox. The first targeted application is Semantic Segmentation for Land Cover Mapping. The Trainer for this task already existed and worked. However, the subsequent segmentation with the trained model and the correct assembly of the individual patches still had to be solved.

## Current status
[TorchGeo_ArcGISPro.pyt](https://github.com/ClaraVth/torchgeo_arcgis/blob/main/TorchGeo_ArcGISPro.pyt) contains the tool and works in the environment of ArcGIS Pro. After the first run, it seems to retain some memory, which eventually causes the program to crash. The first run is successful and creates an output GeoTIFF file with the segmentation result. It recognizes the dominant classes and can assign them to the correct areas. However, smaller classes are suppressed and the shape of the fields is strongly generalized. Further tests to finetune the parameters are necessary.

Another challenge is the usage of the right operating system. Since I have a MacBook, but ArcGIS does not work on MacOS, I have to use a virtual machine. However, there I cannot access the GPU directly. To be able to use the efficiency of the GPU , I have created a separate file [torchgeo_logic_GPU_MacOS.py](https://github.com/ClaraVth/torchgeo_arcgis/blob/main/torchgeo_logic_GPU_MacOS.py) which runs much faster than the original file using the CPU. That simplifies the finetuning task. To implement that into the .pyt file remains to be solved with a Windows PC.

Next steps:
- Replace the prints by logging
- Integrate the usage of a GPU into the .pyt script
- Give the user more options:
  - Select the DL-model (current default: U-Net with ResNet50 backbone)
  - Allow multiple input images and corresponding masks
  - Allow to use an individually pretrained model
  - Add more Trainers like Instance Segmentation or Classification
- Implement details for a better user experience:
  - Automatic Symbolization of the result with the classes from the mask layer
  - Detailed progress bar

## Application
### Set up the environment
ArcGIS Pro needs to be installed. It does not work on macOS! I use version 3.4

Create a virtual environment with Python 3.11 and install the following packages:
```sh
conda create -n NAME python=3.11
```
```sh
conda activate NAME
```
```sh
conda install arcpy=3.4 -c esri
conda install pytorch torchvision  -c pytorch
pip install torchgeo
```

In ArcGIS Pro:
- Set the virtual environment: Go to Settings > Package Manager > Environment Manager > Add existing environment > Activate in list
- Add the .pyt script to the ArcGIS Pro Project Folder
- Add Toolbox: Content Pane > Right-click on Toolboxes > Add Toolbox (Navigate to location on PC)


### Example Usage
I downloaded the Landsat 7, Landsat 8, and CDL datasets for testing from [here](https://huggingface.co/datasets/torchgeo/tutorials/tree/main).\
Larger datasets can be loaded from the [USGS EarthExplorer](https://earthexplorer.usgs.gov) and [USDA](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php).

