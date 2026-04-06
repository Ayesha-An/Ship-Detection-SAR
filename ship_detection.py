# cfar_ship_detection_output_folder_improved.py
import rasterio
import numpy as np
from scipy.ndimage import uniform_filter, label, center_of_mass
import matplotlib.pyplot as plt
import json
import os
from rasterio.transform import xy

# ---------------------------
# Parameters
# ---------------------------
sigma0_file = r"output\preprocessed_image1.tif"       
water_mask_file = r"output\water_mask.tif" 
output_folder = r"output"          
os.makedirs(output_folder, exist_ok=True)

cfar_output_file = os.path.join(output_folder, "cfar_ships.tif")
geojson_output_file = os.path.join(output_folder, "detected_ships.geojson")

window_size = 50       # Local background window (pixels)
threshold_factor = 5.0  # How many std deviations above mean to flag

# ---------------------------
# Load SAR image
# ---------------------------
with rasterio.open(sigma0_file) as src:
    sigma0 = src.read(1).astype(np.float32)
    profile = src.profile
    transform = src.transform
    crs = src.crs  # raster CRS

# ---------------------------
# Optional water mask
# ---------------------------
try:
    with rasterio.open(water_mask_file) as mask_src:
        water_mask = mask_src.read(1).astype(bool)
    sigma0 = np.where(water_mask, sigma0, np.nan)
except FileNotFoundError:
    print("Water mask not found. Using entire image.")
    water_mask = np.ones_like(sigma0, dtype=bool)  # use all pixels

# ---------------------------
# Local background statistics
# ---------------------------
local_mean = uniform_filter(np.nan_to_num(sigma0 * water_mask, nan=0.0), size=window_size, mode='constant')
local_mean_sq = uniform_filter(np.nan_to_num((sigma0 * water_mask)**2, nan=0.0), size=window_size, mode='constant')
local_var = local_mean_sq - local_mean**2
local_std = np.sqrt(np.maximum(local_var, 1e-10))

# ---------------------------
# CFAR detection (ships)
# ---------------------------
cfar_mask = ((sigma0 > (local_mean + threshold_factor * local_std)) & (water_mask == 1)).astype(np.uint8)

# ---------------------------
# Cluster detections
# ---------------------------
labeled, num_features = label(cfar_mask)
print(f"Number of clusters detected: {num_features}")

# ---------------------------
# Extract centroids in pixel coordinates
# ---------------------------
centroids_pixel = center_of_mass(cfar_mask, labeled, range(1, num_features + 1))

# ---------------------------
# Convert pixel coordinates to raster CRS coordinates
# ---------------------------
centroids_crs = []
for r, c in centroids_pixel:
    x, y = xy(transform, int(r), int(c))  # returns coordinates in raster CRS
    centroids_crs.append((x, y))

# ---------------------------
# Create GeoJSON for centroids in raster CRS (same as water mask / raster)
# ---------------------------
features = []
for idx, (x, y) in enumerate(centroids_crs, start=1):
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [x, y]  # keep in raster CRS (projected coordinates)
        },
        "properties": {
            "id": idx
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features,
    "crs": { 
        "type": "name",
        "properties": {
            "name": str(crs)
        }
    }
}

with open(geojson_output_file, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"GeoJSON of detected ship centroids saved to {geojson_output_file}")

# ---------------------------
# Save CFAR mask GeoTIFF
# ---------------------------
profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
with rasterio.open(cfar_output_file, "w", **profile) as dst:
    dst.write(cfar_mask, 1)

print(f"CFAR detection mask saved to {cfar_output_file}")

# ---------------------------
# Optional visualization
# ---------------------------
plt.figure(figsize=(10, 8))
plt.imshow(sigma0, cmap='gray')
plt.imshow(cfar_mask, cmap='Reds', alpha=0.5)
plt.title("CFAR Ship Detections Overlay")
plt.axis('off')
plt.show()