"""
Reads .SAFE files, preprocesses, subsets to AOI and save it as geotiff.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import SNAP Python API
from esa_snappy import ProductIO, GPF, HashMap, jpy
# Get Java types
String = jpy.get_type('java.lang.String')
ArrayList = jpy.get_type('java.util.ArrayList')

# Create output folder
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def geojson_to_wkt(geojson_file):
    """Convert GeoJSON file to WKT string"""
    with open(geojson_file, 'r') as f:
        data = json.load(f)

    # Extract coordinates from GeoJSON
    if 'features' in data:
        coords = data['features'][0]['geometry']['coordinates']
    elif 'geometry' in data:
        coords = data['geometry']['coordinates']
    elif 'coordinates' in data:
        coords = data['coordinates']
    else:
        raise ValueError("Invalid GeoJSON format")

    # Convert to WKT POLYGON format
    polygon_coords = coords[0]  # First ring for polygon
    wkt_coords = ', '.join([f"{lon} {lat}" for lon, lat in polygon_coords])
    wkt = f"POLYGON(({wkt_coords}))"

    return wkt

# --- Helper function to print min/max of a band ---
def print_band_stats(product, band_name):
    """
    Efficiently prints min/max of a band using ESA SNAP snappy + NumPy
    """
    band = product.getBand(band_name)
    width = band.getRasterWidth()
    height = band.getRasterHeight()
    data = np.zeros(width * height, np.float32)

    # Read the full band into the NumPy array
    band.readPixels(0, 0, width, height, data)
    data = data.reshape((height, width))

    # Compute min/max
    min_val = np.min(data)
    max_val = np.max(data)
    print(f"{band_name} -> Min: {min_val:.6f}, Max: {max_val:.6f}")


# ========== CONFIGURATION ==========
# Update these paths to your SAFE files
SAFE_IMAGE1 = r"input/S1A_IW_GRDH_1SDV_20260401T170909_20260401T170934_063889_080915_2132.SAFE"


# AOI GeoJSON file path
AOI_GEOJSON = r"input/ship.geojson"  

# Convert GeoJSON to WKT
AOI_WKT = geojson_to_wkt(AOI_GEOJSON)
print(f"✓ Loaded AOI from: {AOI_GEOJSON}")


def preprocess_safe_to_geotiff(safe_path, output_geotiff, aoi_wkt):
    """
    Preprocess Sentinel-1 SAFE file and save as GeoTIFF
    Returns: VV band as numpy array
    """
    print(f"\nProcessing: {safe_path}")

    # 1. Read SAFE file
    print("  [1/7] Reading SAFE...")
    product = ProductIO.readProduct(safe_path)
    print("Step 1: Original product")

    # Extract and display metadata
    print("\n METADATA:")
    print(f"   Product Name: {product.getName()}")
    print(f"   Product Type: {product.getProductType()}")
    print(f"   Description: {product.getDescription()}")
    print(f"   Scene Size: {product.getSceneRasterWidth()} x {product.getSceneRasterHeight()} pixels")

    # Get CRS information
    geocoding = product.getSceneGeoCoding()
    if geocoding:
        crs = geocoding.getMapCRS()
        if crs:
            print(f"   CRS: {crs.getName()}")
        else:
            print(f"   CRS: Not projected (lat/lon)")

    # List all bands
    print("\n BANDS:")
    band_names = product.getBandNames()
    for i, band_name in enumerate(band_names):
        band = product.getBand(band_name)
        print(f"   [{i+1}] {band_name} - {band.getRasterWidth()}x{band.getRasterHeight()}")

    print()

    # 2. Subset to AOI (EARLY to reduce data volume)
    print("  [2/7] Subsetting to AOI...")
    params = HashMap()
    params.put('geoRegion', aoi_wkt)
    params.put('copyMetadata', True)
    subset = GPF.createProduct('Subset', params, product)
    print(f"Step 2: After Subset - Size: {subset.getSceneRasterWidth()} x {subset.getSceneRasterHeight()}")
    print(f"   Bands: {', '.join(subset.getBandNames())}")

    # 3. Apply orbit file
    print("  [3/7] Applying orbit...")
    params = HashMap()
    params.put('orbitType', 'Sentinel Precise (Auto Download)')
    apply_orbit = GPF.createProduct('Apply-Orbit-File', params, subset)
    print("Step 3: After Orbit File")
    print_band_stats(apply_orbit, 'Intensity_VV')
    print(f"   Bands: {', '.join(apply_orbit.getBandNames())}")

    # 4. Remove thermal noise
    print("  [4/7] Removing thermal noise...")
    params = HashMap()
    params.put('removeThermalNoise', True)
    params.put('reIntroduceThermalNoise', False)
    thermal_noise_removed = GPF.createProduct('ThermalNoiseRemoval', params, apply_orbit)
    print("Step 4: After Thermal Noise Removal")
    print_band_stats(thermal_noise_removed, 'Intensity_VV')
    print(f"   Bands: {', '.join(thermal_noise_removed.getBandNames())}")

    # 5. Calibrate to Sigma0
    print("  [5/7] Calibrating...")
    params = HashMap()
    params.put('outputSigmaBand', True)
    params.put('sourceBands', 'Intensity_VV')
    params.put('selectedPolarisations', 'VV')
    params.put('outputImageScaleInDb', False)
    caliberate = GPF.createProduct('Calibration', params, thermal_noise_removed)
    print("Step 5: After Calibration (Sigma0_VV)")
    print_band_stats(caliberate, 'Sigma0_VV')
    print(f"   Bands: {', '.join(caliberate.getBandNames())}")

    # 6. Speckle filter
    print("  [6/7] Speckle filtering...")
    params = HashMap()
    params.put('filter', 'Lee')
    params.put('windowSize', '5x5')
    speckle_filtered = GPF.createProduct('Speckle-Filter', params, caliberate)
    print("Step 7: After Speckle Filter (Sigma0_VV)")
    print_band_stats(speckle_filtered, 'Sigma0_VV')
    print(f"   Bands: {', '.join(speckle_filtered.getBandNames())}")

    # 7. Terrain correction (final georeferencing)
    print("  [7/7] Terrain correction...")
    params = HashMap()
    params.put('demName', 'SRTM 3Sec')
    params.put('pixelSpacingInMeter', 10.0)
    params.put('mapProjection', 'EPSG:32640')
    product = GPF.createProduct('Terrain-Correction', params, speckle_filtered)
    print_band_stats(product, 'Sigma0_VV')
    print(f"   Bands: {', '.join(product.getBandNames())}")

    # Save as GeoTIFF
    print(f"  Saving GeoTIFF: {output_geotiff}")
    ProductIO.writeProduct(product, str(output_geotiff), 'GeoTIFF')
    print(f"      ✓ Saved successfully!")


def main():
    """Main pipeline"""

    print("\n" + "="*60)
    print("SENTINEL-1 OIL TANK CHANGE DETECTION")
    print("="*60)

    # Preprocess Image to GeoTIFF
    print("\n[STEP 1] Preprocessing Image 1 to GeoTIFF...")
    geotiff1 = OUTPUT_DIR / 'preprocessed_image1.tif'
    vv1 = preprocess_safe_to_geotiff(SAFE_IMAGE1, geotiff1, AOI_WKT)



if __name__ == "__main__":
    main()
