import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# --- Open the GeoTIFF ---
with rasterio.open(r"D:\Master\SAR\ship_detection\output\preprocessed_image1.tif") as src:
    sigma0 = src.read(1).astype(np.float32)  # read first band
    profile = src.profile

# --- Remove invalid/zero values ---
sigma0[sigma0 <= 0] = np.nan

# --- Convert to dB ---
sigma0_db = 10 * np.log10(sigma0)
print(f"Sigma0 stats -> Min: {np.nanmin(sigma0_db):.2f} dB, Max: {np.nanmax(sigma0_db):.2f} dB")

# --- Create water mask ---
# Water has low sigma0 in dB, e.g., < -15 dB
threshold_db = -20
water_mask = sigma0_db <= threshold_db 

print(f"Detected water pixels: {np.sum(water_mask)}")

labeled, num_features = label(water_mask)

min_area = 800  # minimum number of pixels for a water channel, adjust based on resolution
clean_water_mask = np.zeros_like(water_mask, dtype=bool)

for i in range(1, num_features + 1):
    if np.sum(labeled == i) >= min_area:
        clean_water_mask[labeled == i] = True


# --- Save water mask as a new GeoTIFF ---
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw',
    nodata=0
)

with rasterio.open("output/water_mask.tif", "w", **profile) as dst:
    dst.write(clean_water_mask .astype(np.uint8), 1)

print("Water mask saved as water_mask.tif")

# --- Optional: Quick plot ---
plt.imshow(clean_water_mask , cmap='Blues')
plt.title("Water Mask")
plt.axis('off')
plt.show()