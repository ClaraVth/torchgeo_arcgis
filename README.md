# torchgeo_arcgis
Implementation of TorchGeo into ArcGIS Pro

## Set up the enviroment
ArcGIS Pro needs to be installed. It does only work on Windows! I use version 3.4

Create a virtual environment with python 3.11 and install the following packages:
```sh
conda create -n NAME python=3.11
conda activate NAME
conda install arcpy=3.4 -c esri
conda install pytorch torchvision  -c pytorch
pip install torchgeo
```

In ArcGIS Pro:
- Set the virtual environment: Settings > Package Manager > Environment Manager > Add existing environment > Activate in list
- Add the .pyt script to the ArcGIS Pro Project Folder
- Add Toolbox: Content Pane > Right Click on Toolboxes > Add Toolbox (Navigate to location on PC)

## Example Usage
So far, the Tool is designed for Landsat data. Therefore, I downloaded the Landsat 7, Landsat 8, and CDL datasets from https://huggingface.co/datasets/torchgeo/tutorials/tree/main.
