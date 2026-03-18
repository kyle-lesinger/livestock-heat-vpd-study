#!/usr/bin/env python3
"""
Create region and state masks for MERRA-2 grid using point-in-polygon assignment.

This script:
1. Reads MERRA-2 grid coordinates from an example file
2. Creates point geometries for each grid cell centroid
3. Assigns each point to a state using spatial join
4. Creates region/state masks and metadata variables
5. Saves to NetCDF with proper CF conventions
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

# Input shapefile
SHAPEFILE = Path("../shpfiles/cb_2018_us_state_20m.shp")

# Example MERRA-2 file to get grid coordinates
MERRA_EXAMPLE = Path("daily_data/merra2_us_20200615.nc")

# Output file
OUTPUT_FILE = Path("masks/region_mask.nc")

# Region definitions (same as before)
REGION_MAPPING = {
    'Northeast': ['VT', 'NH', 'ME', 'MA', 'RI', 'CT', 'NY', 'PA', 'NJ'],
    'Mid-Atlantic': ['MD', 'DE', 'VA', 'WV'],
    'Atlantic Coast': ['NC', 'SC'],
    'Southeast': ['GA', 'FL', 'AL', 'MS', 'LA', 'AR', 'TN', 'KY'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO'],
    'South Central': ['TX', 'OK'],
    'Central Plains': ['KS', 'NE', 'SD', 'ND'],
    'Northern Rockies': ['MT', 'WY', 'ID'],
    'Southwest Pacific': ['CO', 'UT', 'NV', 'AZ', 'NM'],
    'Pacific Northwest': ['WA', 'OR', 'CA']
}

# Create reverse mapping: state -> region_id
state_to_region_id = {}
state_to_region_name = {}
for region_id, (region_name, states) in enumerate(REGION_MAPPING.items(), start=1):
    for state in states:
        state_to_region_id[state] = region_id
        state_to_region_name[state] = region_name

# =============================================================================
# Load MERRA-2 grid
# =============================================================================

print(f"Loading MERRA-2 grid from: {MERRA_EXAMPLE}")
ds = xr.open_dataset(MERRA_EXAMPLE)
lats = ds['lat'].values
lons = ds['lon'].values
ds.close()

nlat, nlon = len(lats), len(lons)
print(f"Grid dimensions: {nlat} lat × {nlon} lon")
print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}")

# =============================================================================
# Load state boundaries
# =============================================================================

print(f"\nLoading shapefile: {SHAPEFILE}")
states_gdf = gpd.read_file(SHAPEFILE)

# Filter to continental US (exclude AK, HI, PR, territories)
conus_states = states_gdf[~states_gdf['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS'])]
print(f"Loaded {len(conus_states)} CONUS states")

# Ensure CRS is WGS84
if conus_states.crs != 'EPSG:4326':
    conus_states = conus_states.to_crs('EPSG:4326')

# =============================================================================
# Create point geometries for each grid cell
# =============================================================================

print("\nCreating point geometries for grid cells...")
points = []
lat_indices = []
lon_indices = []

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        points.append(Point(lon, lat))
        lat_indices.append(i)
        lon_indices.append(j)

# Create GeoDataFrame of points
points_gdf = gpd.GeoDataFrame({
    'lat_idx': lat_indices,
    'lon_idx': lon_indices
}, geometry=points, crs='EPSG:4326')

print(f"Created {len(points_gdf)} grid cell points")

# =============================================================================
# Spatial join: assign each point to a state
# =============================================================================

print("\nPerforming spatial join (point-in-polygon)...")
joined = gpd.sjoin(
    points_gdf,
    conus_states[['STUSPS', 'NAME', 'geometry']],
    how='left',
    predicate='within'
)

print(f"Matched {joined['STUSPS'].notna().sum()} / {len(joined)} points to states")

# =============================================================================
# Create mask arrays
# =============================================================================

print("\nCreating mask arrays...")

# Initialize arrays (0 = outside US)
region_mask = np.zeros((nlat, nlon), dtype=np.uint8)
state_mask = np.zeros((nlat, nlon), dtype=np.uint8)

# Get unique states and assign IDs
states_in_data = joined['STUSPS'].dropna().unique()
state_id_map = {state: idx for idx, state in enumerate(sorted(states_in_data), start=1)}

# Fill mask arrays
for _, row in joined.iterrows():
    i = row['lat_idx']
    j = row['lon_idx']
    state_abbr = row['STUSPS']

    if pd.notna(state_abbr):
        # Assign state mask
        state_mask[i, j] = state_id_map[state_abbr]

        # Assign region mask
        if state_abbr in state_to_region_id:
            region_mask[i, j] = state_to_region_id[state_abbr]

print(f"Region mask: {(region_mask > 0).sum()} / {region_mask.size} pixels assigned")
print(f"State mask: {(state_mask > 0).sum()} / {state_mask.size} pixels assigned")

# =============================================================================
# Create metadata arrays
# =============================================================================

# Region metadata (10 regions)
region_ids = np.arange(1, 11, dtype=np.uint8)
region_names = [name for name in REGION_MAPPING.keys()]

# State metadata (all states in data)
state_ids_sorted = sorted(state_id_map.items(), key=lambda x: x[1])
state_abbrs = np.array([abbr for abbr, _ in state_ids_sorted], dtype='S2')
state_names_list = [
    conus_states[conus_states['STUSPS'] == abbr]['NAME'].values[0]
    for abbr, _ in state_ids_sorted
]
state_names = np.array(state_names_list, dtype=object)
state_ids_arr = np.array([sid for _, sid in state_ids_sorted], dtype=np.uint8)

# State to region mapping
state_regions = np.array([
    state_to_region_id.get(abbr, 0) for abbr, _ in state_ids_sorted
], dtype=np.uint8)

# =============================================================================
# Create xarray Dataset
# =============================================================================

print("\nCreating NetCDF dataset...")

ds_out = xr.Dataset(
    # Data variables
    data_vars={
        'region_mask': (
            ['lat', 'lon'],
            region_mask,
            {
                'long_name': 'US Region Mask (10 regions)',
                'description': 'Region ID for each grid cell (0=non-US, 1-10=regions)',
                'flag_values': np.arange(11, dtype=np.uint8),
                'flag_meanings': 'non-US ' + ' '.join(region_names),
                '_FillValue': np.uint8(255)
            }
        ),
        'state_mask': (
            ['lat', 'lon'],
            state_mask,
            {
                'long_name': 'US State Mask',
                'description': f'State ID for each grid cell (0=non-US, 1-{len(state_id_map)}=states)',
                '_FillValue': np.uint8(255)
            }
        ),
        'region_id': (
            ['region'],
            region_ids,
            {
                'long_name': 'Region ID',
                'description': 'Unique identifier for each region'
            }
        ),
        'region_name': (
            ['region'],
            region_names,
            {
                'long_name': 'Region Name',
                'description': 'Full name of each region'
            }
        ),
        'state_id': (
            ['state'],
            state_ids_arr,
            {
                'long_name': 'State ID',
                'description': 'Unique identifier for each state'
            }
        ),
        'state_abbr': (
            ['state'],
            state_abbrs,
            {
                'long_name': 'State Abbreviation',
                'description': 'Two-letter state postal code'
            }
        ),
        'state_name': (
            ['state'],
            state_names,
            {
                'long_name': 'State Name',
                'description': 'Full state name'
            }
        ),
        'state_region': (
            ['state'],
            state_regions,
            {
                'long_name': 'State to Region Mapping',
                'description': 'Region ID for each state'
            }
        )
    },
    # Coordinates
    coords={
        'lat': (
            ['lat'],
            lats,
            {
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'standard_name': 'latitude'
            }
        ),
        'lon': (
            ['lon'],
            lons,
            {
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'standard_name': 'longitude'
            }
        ),
        'region': (
            ['region'],
            region_ids,
            {
                'long_name': 'Region ID',
                'description': 'Region dimension coordinate'
            }
        ),
        'state': (
            ['state'],
            state_ids_arr,
            {
                'long_name': 'State ID',
                'description': 'State dimension coordinate'
            }
        )
    },
    # Global attributes
    attrs={
        'title': 'US Region and State Masks for MERRA-2 Grid',
        'description': 'Spatial masks for 10 US regions and individual states aligned to MERRA-2 grid',
        'grid_resolution': '0.5° latitude × 0.625° longitude (approximately 55 km)',
        'grid_shape': f'{nlat} lat × {nlon} lon',
        'projection': 'WGS84 (EPSG:4326)',
        'source_shapefile': 'cb_2018_us_state_20m.shp (US Census Bureau)',
        'creation_date': datetime.now().isoformat(),
        'conventions': 'CF-1.8',
        'method': 'Point-in-polygon assignment of grid cell centroids'
    }
)

# =============================================================================
# Write to NetCDF
# =============================================================================

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print(f"\nWriting to: {OUTPUT_FILE}")

# Only compress numeric variables (not strings)
numeric_vars = ['region_mask', 'state_mask', 'region_id', 'state_id', 'state_region']
encoding = {var: {'zlib': True, 'complevel': 4} for var in numeric_vars if var in ds_out.data_vars}

ds_out.to_netcdf(OUTPUT_FILE, encoding=encoding, format='NETCDF4')

print("\n✓ Complete!")
print(f"\nOutput file: {OUTPUT_FILE}")
print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
print("\nDataset structure:")
print(ds_out)
