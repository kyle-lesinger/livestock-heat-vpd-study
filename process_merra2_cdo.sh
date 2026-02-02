#!/bin/bash
################################################################################
# MERRA-2 CDO Processing Script
#
# This script processes a single MERRA-2 NetCDF file using CDO commands.
# It's designed to be called from Python or run standalone.
#
# Usage: ./process_merra2_cdo.sh INPUT_FILE OUTPUT_FILE BBOX
#   INPUT_FILE: Path to downloaded MERRA-2 NetCDF file
#   OUTPUT_FILE: Path for output file
#   BBOX: Bounding box as "lonmin,lonmax,latmin,latmax"
#
# Example:
#   ./process_merra2_cdo.sh merra2_raw.nc merra2_us_20230601.nc "-125,-66,24,49"
################################################################################

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Wrong number of arguments"
    echo "Usage: $0 INPUT_FILE OUTPUT_FILE BBOX"
    echo "Example: $0 input.nc output.nc \"-125,-66,24,49\""
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
BBOX="$3"

# Check if CDO is installed
if ! command -v cdo &> /dev/null; then
    echo "Error: CDO is not installed!"
    echo "Install with: brew install cdo  (or conda install -c conda-forge cdo)"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Processing with CDO..."
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Bbox: $BBOX"

# Step 1: Subset region and select variables
SUBSET_FILE="$TEMP_DIR/subset.nc"
cdo -f nc4 -z zip_4 \
    -sellonlatbox,$BBOX \
    -selname,T2M,QV2M,PS \
    "$INPUT_FILE" \
    "$SUBSET_FILE"

# Step 2: Calculate VPD and convert T2M to Celsius
# VPD Formula:
#   T2M_C = T2M - 273.15
#   es = 0.6108 * exp((17.27 * T2M_C) / (T2M_C + 237.3))
#   ea = (QV2M * PS) / (0.622 + 0.378 * QV2M) / 1000
#   VPD = es - ea

VPD_EXPR="T2M_C=T2M-273.15;es=0.6108*exp((17.27*T2M_C)/(T2M_C+237.3));ea=(QV2M*PS)/(0.622+0.378*QV2M)/1000;VPD=es-ea;"

CALC_FILE="$TEMP_DIR/calc.nc"
cdo -f nc4 -z zip_4 \
    -expr,"$VPD_EXPR" \
    "$SUBSET_FILE" \
    "$CALC_FILE"

# Step 3: Rename T2M_C to T2M, then select only T2M and VPD
# Note: CDO operations are chained right-to-left
FINAL_FILE="$TEMP_DIR/final.nc"
cdo -f nc4 -z zip_4 \
    -selname,T2M,VPD \
    -chname,T2M_C,T2M \
    "$CALC_FILE" \
    "$FINAL_FILE"

# Step 4: Convert to float32 for smaller file size (if ncks available)
if command -v ncks &> /dev/null; then
    echo "Converting to float32 with NCO..."
    ncks -O -4 --deflate 4 --ppc default=5 \
        "$FINAL_FILE" \
        "$OUTPUT_FILE"
else
    echo "NCO not available, skipping float32 conversion"
    cp "$FINAL_FILE" "$OUTPUT_FILE"
fi

# Get file size
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "✓ Processing complete! Output: $OUTPUT_FILE ($FILE_SIZE)"
