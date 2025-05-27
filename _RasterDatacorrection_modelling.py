import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from glob import glob

# Set paths
master_raster_path = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/masterdata/3UKLST2020.tif"
input_rasters_folder = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/firelabel_raster/"
output_rasters_folder = "D:/Fire_Data/_Project_ForestFire_LCC/01_LCC_DATA/ModelTestData/output2"

# Create output folder if it doesn't exist
os.makedirs(output_rasters_folder, exist_ok=True)

# Read master raster to get specs
with rasterio.open(master_raster_path) as master:
    master_crs = master.crs
    master_transform = master.transform
    master_width = master.width
    master_height = master.height
    master_dtype = master.dtypes[0]

# Process each raster
for raster_path in glob(os.path.join(input_rasters_folder, "*.tif")):
    raster_name = os.path.basename(raster_path)
    output_path = os.path.join(output_rasters_folder, raster_name)

    with rasterio.open(raster_path) as src:
        # Prepare metadata
        profile = src.profile.copy()
        profile.update({
            'crs': master_crs,
            'transform': master_transform,
            'width': master_width,
            'height': master_height,
            'dtype': master_dtype
        })

        # Open destination raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.bilinear  # You can change to nearest, cubic, etc.
                )

print("✅ All rasters have been successfully resampled and aligned!")
