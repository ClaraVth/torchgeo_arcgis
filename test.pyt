import matplotlib.pyplot as plt

in_image = "torchgeo_arcgis/data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF"
in_mask = "torchgeo_arcgis/data/2023_30m_cdls.tif"
out_folder = "."
batch_size = 8
epochs = 10

img = plt.imread(in_mask)
print(img[:,:,0]-img[:,:,3])
print(img.shape)