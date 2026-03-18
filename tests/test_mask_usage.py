#!/usr/bin/env python3
"""
Quick test to verify region_mask.nc can be used for filtering MERRA-2 data
"""

import xarray as xr
import numpy as np

print("="*70)
print("Testing region_mask.nc with MERRA-2 data")
print("="*70)

# Load mask
print("\n1. Loading region_mask.nc...")
mask_ds = xr.open_dataset('region_mask.nc')
print(f"   ✓ Loaded. Shape: {mask_ds.region_mask.shape}")

# Load sample MERRA-2 data
print("\n2. Loading sample MERRA-2 data...")
data_ds = xr.open_dataset('daily_data/merra2_us_20200615.nc')
t2m = data_ds.T2M.mean(dim='time')
print(f"   ✓ Loaded. Shape: {t2m.shape}")

# Test region filtering
print("\n3. Testing region filtering (Region 6: South Central)...")
region_id = 6
t2m_region = t2m.where(mask_ds.region_mask == region_id)
region_pixels = np.sum(~np.isnan(t2m_region.values))
region_mean = t2m_region.mean().values
print(f"   ✓ Filtered {region_pixels} pixels")
print(f"   ✓ Mean temperature: {region_mean:.2f} °C")

# Test state filtering
print("\n4. Testing state filtering (Texas)...")
tx_idx = np.where(mask_ds.state_abbr.values == 'TX')[0][0]
tx_id = mask_ds.state_id.values[tx_idx]
t2m_tx = t2m.where(mask_ds.state_mask == tx_id)
tx_pixels = np.sum(~np.isnan(t2m_tx.values))
tx_mean = t2m_tx.mean().values
print(f"   ✓ Texas state_id: {tx_id}")
print(f"   ✓ Filtered {tx_pixels} pixels")
print(f"   ✓ Mean temperature: {tx_mean:.2f} °C")

# Test multiple regions
print("\n5. Testing all 10 regions...")
for rid in range(1, 11):
    region_name = mask_ds.region_name.values[rid - 1]
    region_data = t2m.where(mask_ds.region_mask == rid)
    pixels = np.sum(~np.isnan(region_data.values))
    if pixels > 0:
        mean_temp = region_data.mean().values
        print(f"   Region {rid:2d} ({region_name:20s}): {pixels:4d} pixels, T2M = {mean_temp:.2f}°C")
    else:
        print(f"   Region {rid:2d} ({region_name:20s}): No pixels")

print("\n" + "="*70)
print("All tests passed! ✓")
print("="*70)

mask_ds.close()
data_ds.close()
