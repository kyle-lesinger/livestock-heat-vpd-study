"""
MERRA-2 Data Processing Functions

This module provides functions for downloading, processing, and saving
MERRA-2 hourly temperature and VPD data for the United States.

Author: Generated for VEDA Stories - Livestock and Heat
Date: 2026-02-02
"""

import earthaccess
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def calculate_vpd(temperature_k, specific_humidity, surface_pressure):
    """
    Calculate Vapor Pressure Deficit (VPD) using Tetens formula.

    This function performs lazy computation - no actual data is computed
    until the result is saved or explicitly computed.

    Parameters
    ----------
    temperature_k : xarray.DataArray
        2-meter air temperature in Kelvin
    specific_humidity : xarray.DataArray
        2-meter specific humidity in kg/kg
    surface_pressure : xarray.DataArray
        Surface pressure in Pa

    Returns
    -------
    vpd : xarray.DataArray
        Vapor pressure deficit in kPa with proper metadata

    Notes
    -----
    VPD = es - ea
    where:
        es = saturation vapor pressure (from Tetens formula)
        ea = actual vapor pressure (from specific humidity)
    """
    # Convert temperature to Celsius
    temp_c = temperature_k - 273.15

    # Calculate saturation vapor pressure (es) in kPa using Tetens formula
    # es = 0.6108 * exp((17.27 * T) / (T + 237.3))
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

    # Calculate actual vapor pressure (ea) from specific humidity
    # ea = (q * P) / (0.622 + 0.378 * q)
    ea = (specific_humidity * surface_pressure) / (0.622 + 0.378 * specific_humidity)
    ea = ea / 1000  # Convert from Pa to kPa

    # Calculate VPD
    vpd = es - ea

    # Add metadata
    vpd.attrs['long_name'] = 'Vapor Pressure Deficit'
    vpd.attrs['units'] = 'kPa'
    vpd.attrs['description'] = 'Calculated from T2M, QV2M, and PS using Tetens formula'
    vpd.attrs['formula'] = 'VPD = es(T2M) - ea(QV2M, PS)'

    return vpd


def check_file_exists(date, output_dir):
    """
    Check if a processed file already exists for the given date.

    Parameters
    ----------
    date : datetime or str
        Date to check (YYYY-MM-DD format if string)
    output_dir : str or Path
        Directory where processed files are stored

    Returns
    -------
    bool
        True if file exists, False otherwise
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    output_dir = Path(output_dir)
    filename = f"merra2_us_{date.strftime('%Y%m%d')}.nc"
    filepath = output_dir / filename

    return filepath.exists()


def get_us_subset(ds, bbox):
    """
    Apply spatial subsetting to extract US region.

    This function performs lazy subsetting - no data is loaded into memory.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with global coverage
    bbox : tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    ds_subset : xarray.Dataset
        Spatially subsetted dataset (lazy)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    ds_subset = ds.sel(
        lon=slice(min_lon, max_lon),
        lat=slice(min_lat, max_lat)
    )

    return ds_subset


def process_single_day(date, bbox, collection_id, output_dir, auth=None):
    """
    Process a single day of MERRA-2 data: download, calculate VPD, and save.

    This function:
    1. Checks if file already exists (skips if so)
    2. Searches for MERRA-2 granule for the date
    3. Opens and subsets data (lazy operations)
    4. Calculates VPD
    5. Creates output dataset with T2M (°C) and VPD (kPa)
    6. Saves to NetCDF (triggers computation only once)

    Parameters
    ----------
    date : datetime or str
        Date to process (YYYY-MM-DD format if string)
    bbox : tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    collection_id : str
        MERRA-2 collection short name (e.g., 'M2T1NXSLV')
    output_dir : str or Path
        Directory to save processed files
    auth : earthaccess auth object, optional
        Authentication object. If None, will attempt to login.

    Returns
    -------
    dict
        Status dictionary with 'success', 'message', and 'filepath' keys
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if check_file_exists(date, output_dir):
        return {
            'success': True,
            'message': 'File already exists, skipped',
            'filepath': output_dir / f"merra2_us_{date.strftime('%Y%m%d')}.nc",
            'skipped': True
        }

    try:
        # Authenticate if not provided
        if auth is None:
            auth = earthaccess.login()

        # Search for granules for this specific date
        date_str = date.strftime('%Y-%m-%d')
        next_day = (date + timedelta(days=1)).strftime('%Y-%m-%d')

        results = earthaccess.search_data(
            short_name=collection_id,
            bounding_box=bbox,
            temporal=(date_str, next_day),
        )

        if len(results) == 0:
            return {
                'success': False,
                'message': f'No granules found for {date_str}',
                'filepath': None,
                'skipped': False
            }

        # Open the file (lazy load)
        files = earthaccess.open(results)
        ds = xr.open_mfdataset(files, combine='by_coords')

        # Select only needed variables
        required_vars = ['T2M', 'QV2M', 'PS']
        missing_vars = [v for v in required_vars if v not in ds.data_vars]
        if missing_vars:
            return {
                'success': False,
                'message': f'Missing required variables: {missing_vars}',
                'filepath': None,
                'skipped': False
            }

        ds_vars = ds[required_vars]

        # Apply spatial subset (lazy)
        ds_us = get_us_subset(ds_vars, bbox)

        # Calculate VPD (lazy)
        vpd = calculate_vpd(ds_us['T2M'], ds_us['QV2M'], ds_us['PS'])

        # Convert T2M to Celsius (lazy)
        t2m_celsius = ds_us['T2M'] - 273.15
        t2m_celsius.attrs['long_name'] = '2-meter air temperature'
        t2m_celsius.attrs['units'] = 'degrees_Celsius'
        t2m_celsius.attrs['standard_name'] = 'air_temperature'
        t2m_celsius.attrs['original_units'] = 'K'

        # Create output dataset with only T2M and VPD
        ds_out = xr.Dataset({
            'T2M': t2m_celsius,
            'VPD': vpd
        })

        # Add global attributes
        ds_out.attrs['title'] = 'MERRA-2 Hourly Temperature and VPD for US Lower 48'
        ds_out.attrs['source'] = f'MERRA-2 {collection_id}'
        ds_out.attrs['bbox'] = str(bbox)
        ds_out.attrs['spatial_coverage'] = 'US Lower 48 States'
        ds_out.attrs['temporal_coverage'] = date_str
        ds_out.attrs['processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        ds_out.attrs['institution'] = 'NASA GSFC'
        ds_out.attrs['processed_by'] = 'VEDA Stories - Livestock and Heat Analysis'

        # Save to NetCDF (this triggers the computation)
        output_file = output_dir / f"merra2_us_{date.strftime('%Y%m%d')}.nc"

        # Use compression and float32 to minimize file size
        encoding = {
            'T2M': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': -9999.0
            },
            'VPD': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': -9999.0
            }
        }

        ds_out.to_netcdf(output_file, encoding=encoding)

        # Close datasets
        ds_out.close()
        ds.close()

        return {
            'success': True,
            'message': f'Successfully processed {date_str}',
            'filepath': output_file,
            'skipped': False
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing {date_str}: {str(e)}',
            'filepath': None,
            'skipped': False
        }


def save_daily_data(ds_subset, date, output_dir):
    """
    Save processed daily data to NetCDF file.

    Parameters
    ----------
    ds_subset : xarray.Dataset
        Dataset containing T2M and VPD variables
    date : datetime
        Date of the data
    output_dir : str or Path
        Directory to save the file

    Returns
    -------
    Path
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"merra2_us_{date.strftime('%Y%m%d')}.nc"
    filepath = output_dir / filename

    # Use compression and float32 to minimize file size
    encoding = {
        'T2M': {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': -9999.0
        },
        'VPD': {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': -9999.0
        }
    }

    ds_subset.to_netcdf(filepath, encoding=encoding)

    return filepath


def get_date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date.

    Parameters
    ----------
    start_date : str or datetime
        Start date (YYYY-MM-DD format if string)
    end_date : str or datetime
        End date (YYYY-MM-DD format if string)

    Returns
    -------
    list of datetime
        List of dates to process
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    return pd.date_range(start_date, end_date, freq='D').tolist()
