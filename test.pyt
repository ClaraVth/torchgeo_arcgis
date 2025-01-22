import rasterio

#in_image = "torchgeo_arcgis/data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF"
in_mask = "data/2023_30m_cdls.tif"
out_folder = "."
batch_size = 8
epochs = 10

img = rasterio.open(in_mask)
x =img.read()
#print(x[:,:,0])
print(x.shape)