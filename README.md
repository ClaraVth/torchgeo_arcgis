# torchgeo_arcgis
Implementation of TorchGeo into ArcGIS Pro

# Set up the enviroment
ArcGIS Pro needs to be installed - I use version 3.4 (does only work on Windows)

Create a virtual environment with python 3.11
install the following packages:
- arcpy 3.4
- torchgeo 0.6.2
- torch 2.2.2, torchvision 0.17.2, torchaudio 2.5.0

In ArcGIS:
- Set the virtual environment: Settings > Package Manager > Environment Manager > Add existing environment > Activate in list
- Add the .pyt script to the ArcGIS Pro Project Folder
- Add Toolbox: Content Pane > Right Click on Toolboxes > Add Toolbox (Navigate to location on PC)
- Use Toolbox by filling with layers and input further parameters like batch size
