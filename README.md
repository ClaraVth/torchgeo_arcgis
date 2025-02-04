# torchgeo_arcgis
Implementation of TorchGeo into ArcGIS Pro

## Set up the environment
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

## Current status
The .pyt file has been created and works in the environment of ArcGIS Pro. After the first run, it seems to retain some memory, which eventually causes the program to crash. Therefore, I have isolated the code for semantic segmentation into the torchgeo_logic.py file. So far, I can run the script successfully and obtain an output GeoTIFF file with the segmentation. It recognizes the dominant classes and can assign them to the correct areas. However, smaller classes are suppressed and the shape of the fields is strongly generalized.\\

In order to use the GPU on my MacBook, I created a new file [torchgeo_logic_GPU_Mac_stitched.py]() which runs much faster than the original file using the CPU. To implement that into the .pyt file keeps to be solved with a Windows PC.

## Example Usage
I downloaded the Landsat 7, Landsat 8, and CDL datasets for testing from [here](https://huggingface.co/datasets/torchgeo/tutorials/tree/main).\
Larger datasets can be loaded from the [USGS EarthExplorer](https://earthexplorer.usgs.gov) and [USDA](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php).

