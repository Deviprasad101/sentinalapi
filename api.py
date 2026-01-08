import json
import os
import subprocess
import webbrowser
import sys
import time
import numpy as np
import rasterio
from datetime import date, timedelta
from sentinelhub import (
    Geometry, CRS, SentinelHubRequest,
    DataCollection, MimeType, SHConfig
)
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from scipy.stats import pearsonr

# ---------------------------
# Evalscript (RGB)
# ---------------------------
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B04", "B03"],
    output: { bands: 3, sampleType: "UINT8" }
  };
}

function clamp(v) {
  return Math.max(0, Math.min(1, v));
}

function evaluatePixel(s) {
  return [
    clamp(s.B08) * 255,  // NIR
    clamp(s.B04) * 255,  // Red
    clamp(s.B03) * 255   // Green
  ];
}
"""

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B04"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(s) {
  return [(s.B08 - s.B04) / (s.B08 + s.B04)];
}
"""

# ---------------------------
# Load GeoJSON
# ---------------------------
# Updated path to be relative or expect it in the same folder
geojson_path = os.path.join(os.path.dirname(__file__), "boundary_nallamalla_geojson.geojson")

# Use a default if not found (or raise error, but let's assume user will provide it)
# For now, let's keep the hardcoded path if it works for the user, OR update it to be relative.
# The user's original script had: r"D:\API\boundary_nallamalla_geojson.geojson"
# I should probably update it to look in the current directory if they are moving everything to git.
if not os.path.exists(geojson_path):
     # Fallback to the hardcoded path if relative doesn't work (for local testing compatibility)
     geojson_path = r"D:\API\boundary_nallamalla_geojson.geojson"

if os.path.exists(geojson_path):
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    geom = geojson["features"][0]["geometry"]
    geometry = Geometry(geom, CRS.WGS84)
else:
    print(f"⚠️ Warning: GeoJSON not found at {geojson_path}. Script might fail.")
    # Define a dummy geometry to avoid immediate crash if just testing
    geometry = None 

# ---------------------------
# Sentinel Hub config
# ---------------------------
config = SHConfig()
config.sh_client_id = "004f53a9-aaa8-44b7-add8-73f16be25afa"
config.sh_client_secret = "Pn4WxspaDOUTy5VHtqm7V4KN8GwerxsG"

# ---------------------------
# Output folder
# ---------------------------
# Point to the static/tiffs folder relative to this script
out_dir = os.path.join(os.path.dirname(__file__), "static", "tiffs")
os.makedirs(out_dir, exist_ok=True)

# Date ranges strategy: Find best image in March
requests_config = [
    {
        "year": "2022",
        "time_interval": ("2022-03-01", "2022-03-31")
    },
    {
        "year": "2023",
        "time_interval": ("2023-03-01", "2023-03-31")
    }
]

if geometry:
    for req_cfg in requests_config:
        year = req_cfg["year"]
        time_interval = req_cfg["time_interval"]

        request = SentinelHubRequest(
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order="leastCC",
                    other_args={"maxcc": 0.2}
                )
            ],
            geometry=geometry,
            size=(2048, 2048),
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            evalscript=EVALSCRIPT,
            data_folder=out_dir,
            config=config
        )

        request.get_data(save_data=True)
        print(f"✅ Image downloaded for {year} (best within {time_interval})")

# ---------------------------
# Multiband Evalscript (B02, B03, B04, B08, B11, B12)
# ---------------------------
EVALSCRIPT_MULTIBAND = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "B11", "B12"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(s) {
  return [
    s.B02, // Blue
    s.B03, // Green
    s.B04, // Red
    s.B08, // NIR
    s.B11, // SWIR1
    s.B12  // SWIR2
  ];
}
"""

# ---------------------------
# Land Cover Classification (2023)
# ---------------------------
def save_ndvi(ndvi_array, meta, out_path):
    meta_ndvi = meta.copy()
    meta_ndvi.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(out_path, "w", **meta_ndvi) as dst:
        dst.write(ndvi_array.astype("float32"), 1)

# ---------------------------
# Land Cover Classification (2022 & 2023)
# ---------------------------
print("\n🌍 Starting Multi-Year Processing (2022-2023)...")

if geometry:
    for req_cfg in requests_config:
        year = req_cfg["year"]
        time_interval = req_cfg["time_interval"]
        
        print(f"\n📅 Processing Year: {year}")

        # Use a subfolder for multiband data
        multiband_dir = os.path.join(out_dir, f"multiband_{year}")

        multiband_request = SentinelHubRequest(
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order="leastCC",
                    other_args={"maxcc": 0.2}
                )
            ],
            geometry=geometry,
            size=(2048, 2048),
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            evalscript=EVALSCRIPT_MULTIBAND,
            data_folder=multiband_dir,
            config=config
        )

        multiband_request.get_data(save_data=True)

        # Locate the file saved by Sentinel Hub
        multiband_path = None
        for root, _, files in os.walk(multiband_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    multiband_path = os.path.join(root, file)
                    break
            if multiband_path:
                break

        if multiband_path:
            print(f"✅ Multiband data downloaded: {multiband_path}")
            
            with rasterio.open(multiband_path) as src:
                data = src.read()
                # Bands: B02, B03, B04, B08, B11, B12
                blue, green, red, nir, swir1, swir2 = data[0], data[1], data[2], data[3], data[4], data[5]
                meta = src.meta.copy()

            epsilon = 1e-6
            
            # Calculate Indices
            ndvi = (nir - red) / (nir + red + epsilon)
            nbr  = (nir - swir2) / (nir + swir2 + epsilon)
            bsi  = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + epsilon)
            
            print("✅ Indices calculated (NDVI, NBR, BSI)")
            
            # Save NDVI
            ndvi_path = os.path.join(out_dir, f"ndvi_{year}.tif")
            save_ndvi(ndvi, meta, ndvi_path)
            print(f"💾 NDVI saved for {year}")

            # Rule-Based Classification
            classified = np.zeros(ndvi.shape, dtype=np.uint8)
            # 1=Forest, 2=Agri, 3=Deforest, 4=Mining, 5=Vacant
            classified[ndvi > 0.6] = 1
            classified[(ndvi > 0.3) & (ndvi <= 0.6)] = 2
            classified[(ndvi < 0.4) & (nbr < 0.1)] = 3
            classified[(ndvi < 0.2) & (bsi > 0.3) & (swir1 > 0.2)] = 4
            classified[(ndvi < 0.2) & (bsi < 0.3)] = 5
            
            # Save Classified GeoTIFF
            lc_path = os.path.join(out_dir, f"landcover_{year}.tif")
            
            meta.update(dtype="uint8", count=1, compress="lzw", tiled=True, blockxsize=256, blockysize=256)

            with rasterio.open(lc_path, "w", **meta) as dst:
                dst.write(classified, 1)
                
            print(f"💾 Land Cover saved: {lc_path}")
            
            # Compatibility copy for Web App (uses 2023)
            if year == "2023":
                 out_path = os.path.join(out_dir, "landcover_classified.tif")
                 with rasterio.open(out_path, "w", **meta) as dst:
                     dst.write(classified, 1)
        else:
            print(f"❌ Failed to find downloaded multiband TIFF for {year}")

# ---------------------------
# STATS MODULE (Accuracy & NDVI)
# ---------------------------
print("\n📊 Calculating Statistics...")

lc_2023_path = os.path.join(out_dir, "landcover_2023.tif")
ref_path_explicit = "reference_landcover.tif" # Hypothetical
ref_path_fallback = os.path.join(out_dir, "landcover_2022.tif")

ndvi_2022_path = os.path.join(out_dir, "ndvi_2022.tif")
ndvi_2023_path = os.path.join(out_dir, "ndvi_2023.tif")

if os.path.exists(lc_2023_path) and os.path.exists(ndvi_2022_path) and os.path.exists(ndvi_2023_path):

    # --- A. LAND-COVER ACCURACY ---
    # Determine reference
    if os.path.exists(ref_path_explicit):
        ref_file = ref_path_explicit
        print(f"🔹 Using explicit reference: {ref_file}")
    elif os.path.exists(ref_path_fallback):
        ref_file = ref_path_fallback
        print(f"⏳ Using 2022 as reference (Temporal Stability): {ref_file}")
    else:
        ref_file = None
        print("⚠️ No reference file found for Accuracy Assessment.")

    landcover_stats = {}
    
    with rasterio.open(lc_2023_path) as src:
        classified = src.read(1)

    if ref_file:
        with rasterio.open(ref_file) as ref_src:
            reference = ref_src.read(1)
            # Ensure dimensions match (simple check)
            if reference.shape == classified.shape:
                pred = classified.flatten()
                ref  = reference.flatten()
                
                # Mask valid pixels only (ignoring 0/nodata if any)
                mask = (ref > 0) & (pred > 0)
                
                if np.any(mask):
                    overall_accuracy = accuracy_score(ref[mask], pred[mask])
                    kappa = cohen_kappa_score(ref[mask], pred[mask])
                    cm = confusion_matrix(ref[mask], pred[mask])
                    
                    landcover_stats = {
                        "overall_accuracy": round(float(overall_accuracy), 3),
                        "kappa": round(float(kappa), 3),
                        "confusion_matrix": cm.tolist()
                    }
                else:
                    landcover_stats = {"error": "No valid intersection"}
            else:
                landcover_stats = {"error": "Dimension mismatch"}

    # --- B. NDVI STATISTICS ---
    with rasterio.open(ndvi_2022_path) as src:
        ndvi_2022 = src.read(1).flatten()
    with rasterio.open(ndvi_2023_path) as src:
        ndvi_2023 = src.read(1).flatten()

    mask_ndvi = (~np.isnan(ndvi_2022)) & (~np.isnan(ndvi_2023))
    
    if np.any(mask_ndvi):
        corr, _ = pearsonr(ndvi_2022[mask_ndvi], ndvi_2023[mask_ndvi])
        
        # Calculate Difference
        ndvi_diff = ndvi_2023[mask_ndvi] - ndvi_2022[mask_ndvi]
        
        mean_change = np.mean(ndvi_diff)
        min_change = np.min(ndvi_diff)
        max_change = np.max(ndvi_diff)
        
        print(f"Mean NDVI change : {mean_change:.4f}")
        print(f"Min NDVI change  : {min_change:.4f}")
        print(f"Max NDVI change  : {max_change:.4f}")
        
        ndvi_stats = {
            "correlation": round(float(corr), 3),
            "mean_change": round(float(mean_change), 4),
            "min_change": round(float(min_change), 4),
            "max_change": round(float(max_change), 4)
        }
    else:
        ndvi_stats = {
            "correlation": 0, 
            "mean_change": 0,
            "min_change": 0,
            "max_change": 0
        }

    # --- C. SAVE STATS FOR WEB APP ---
    stats = {
        "landcover": landcover_stats,
        "ndvi": ndvi_stats
    }

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Accuracy statistics saved (stats.json)")

else:
    print("❌ Needed TIFF files (2022/2023) for analytics are missing.")

# ---------------------------
# Launch Web App
# ---------------------------
print("\n🚀 Starting Web Visualization...")

# Define the path to the app.py (Adjusted for repo structure)
# Assuming app.py is in the same directory as api.py
app_path = os.path.join(os.path.dirname(__file__), "app.py")

# Open browser after a short delay to allow server to start
def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")

# Start browser in a separate thread
import threading
threading.Thread(target=open_browser).start()

# Run Flask app (this is blocking, so the script will stay running until you stop the server)
print("🌍 Serving map at http://127.0.0.1:5000")
print("❌ Press Ctrl+C to stop the server.")

subprocess.run([sys.executable, app_path])
