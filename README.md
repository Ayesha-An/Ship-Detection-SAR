# SAR Ship Detection

Automated ship detection pipeline using Sentinel-1 SAR imagery and CFAR (Constant False Alarm Rate) algorithm.

## Overview

This project processes Sentinel-1 GRD imagery to automatically detect ships at sea, combining SAR preprocessing with statistical target detection methods.

## Pipeline

1. **SAR Preprocessing** ([snap_preprocess.py](snap_preprocess.py))
   - Orbit correction, thermal noise removal, and radiometric calibration
   - Speckle filtering and terrain correction
   - AOI subsetting and GeoTIFF export

2. **Water Mask Generation** ([watermask.py](watermask.py))
   - Extracts water bodies using backscatter thresholding
   - Filters small artifacts

3. **Ship Detection** ([ship_detection.py](ship_detection.py))
   - CFAR-based detection with local adaptive thresholding
   - Cluster analysis and centroid extraction
   - GeoJSON export for GIS integration

## Output

- `preprocessed_image1.tif` - Calibrated Sigma0 backscatter
- `water_mask.tif` - Binary water mask
- `cfar_ships.tif` - Ship detection mask
- `detected_ships.geojson` - Ship locations with coordinates

## Dependencies

- ESA SNAP Python API (esa_snappy)
- rasterio, numpy, scipy, matplotlib

## Detection Method

CFAR algorithm detects bright targets (ships) by comparing pixel values to local background statistics, using adaptive mean and standard deviation thresholds in sliding windows.
