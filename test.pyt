import arcpy

arcpy.ImportToolbox(r"C:\ArcGIS_Projects\Torchgeo_ArcGIS\Torchgeo_ArcGIS.atbx", "torchgeo_landuse")

# Verfügbare Tools anzeigen
print("Verfügbare Tools:", arcpy.ListTools("torchgeo_landuse*"))
print("Toolbox importiert:", "torchgeo_landuse" in dir(arcpy))

# Parameter definieren
in_image = r"C:\ArcGIS_Projects\Torchgeo_ArcGIS\LE07_L2SP_022032_20230725_20230820_02_T1_SR_B7.TIF"
in_mask = r"C:\ArcGIS_Projects\Torchgeo_ArcGIS\2023_30m_cdls.tif"
out_folder = r"C:\ArcGIS_Projects\Torchgeo_ArcGIS\LE07_L2SP_0220_TrainLandUseM3"
batch_size = 8
epochs = 10

try:
    arcpy.torchgeo_landuse.TrainLandUseModel(
        in_image=in_image,
        in_mask=in_mask,
        out_folder=out_folder,
        batch_size=batch_size,
        epochs=epochs
    )
except arcpy.ExecuteError:
    print("ArcPy-Fehler:", arcpy.GetMessages(2))
except Exception as e:
    print("Allgemeiner Fehler:", str(e))
else:
    print("Tool erfolgreich ausgeführt.")



