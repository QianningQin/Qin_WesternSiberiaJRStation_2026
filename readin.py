# -*- coding: utf-8 -*-
"""
Functions Importing Data
"""

import pandas as pd
import numpy as np
# import datetime as dt
from datetime import datetime

import os
import gc
from pathlib import Path

import h5py
import glob
import dask
import xarray as xr
from geopy.distance import geodesic
import analysis as ana

## JR-Station
def read_in_tower(file_path, site, skiprow, co2_threshold=800.0, ch4_threshold=8000.0, ap_threshold=1500.0):
    df_list = []
    
    # 1. Load data
    for filename in Path(file_path).glob(site):
        df_temp = pd.read_fwf(filename, skiprows=skiprow)
        df_list.append(df_temp)
    
    if not df_list:
        return pd.DataFrame()

    # 2. Prevent Chained Indexing & Memory Fragmentation
    tower = pd.concat(df_list, axis=0, ignore_index=True).copy()
    df_list.clear()
    gc.collect()
    
    # 3. Vectorized Datetime Extraction
    dt_str = (tower['YEAR'].astype(str) + '-' + 
              tower['MONTH'].astype(str) + '-' + 
              tower['DAY'].astype(str) + '-' + 
              tower['HOUR'].astype(str))
    
    dt_series = pd.to_datetime(dt_str, format='%Y-%m-%d-%H')
    tower['Date'] = dt_series.dt.date
    tower['Hour'] = dt_series.dt.hour
    
    if 'HOUR' in tower.columns:
        tower.drop(columns=['HOUR'], inplace=True)
    
    # 4. Global Numeric Conversion
    cols_to_convert = [c for c in tower.columns if c != 'Date']
    for col in cols_to_convert:
        tower[col] = pd.to_numeric(tower[col], errors='coerce')
    
    # 5. Vectorized Pressure (ap) Logic via Mathematics
    if 'ap' in tower.columns:
        mask_ap_fix = (tower['ap'] >= 0) & (tower['ap'] < 500)
        tower.loc[mask_ap_fix, 'ap'] += 1000.0
    
    # 6. Target Gas Threshold Filtering
    if 'CO2' in tower.columns:
        tower.loc[tower['CO2'] > co2_threshold, 'CO2'] = np.nan
    if 'CH4' in tower.columns:
        tower.loc[tower['CH4'] > ch4_threshold, 'CH4'] = np.nan
    if 'ap' in tower.columns:
        tower.loc[tower['ap'] > ap_threshold, 'ap'] = np.nan
            
    # 7. Downcasting
    int_cols = ['MINUTE', 'Hour', 'YEAR', 'MONTH', 'DAY']
    for col in cols_to_convert:
        if col in int_cols and col in tower.columns:
            tower[col] = tower[col].fillna(0).astype(np.int32)
        else:
            tower[col] = tower[col].astype(np.float64)
            
    return tower.sort_values(['Date', 'Hour', 'MINUTE']).reset_index(drop=True)

  
## Aircraft
def read_in_aircraft(df_aircraft, timezone, strptime, CH4_choice, CO_choice, Temp_choice):
    # 1. Vectorized Cleanup
    # We work on a copy to avoid SettingWithCopy warnings if df_aircraft was a slice
    df = df_aircraft.dropna(subset=['CO2']).loc[df_aircraft['Flag'] != 1].copy()
    df['Alt_Rounded'] = df['Altitude'].round(1)
    df['Altitude'] = df['Alt_Rounded'] / 1000
    
    # 2. Vectorized Datetime Processing
    # pd.to_datetime is significantly faster than strptime in a lambda
    dt_series = pd.to_datetime(df['DateTime'], format=strptime)
    
    # Localize and convert to UTC in one vectorized flow
    dt_utc = dt_series.dt.tz_localize(timezone, ambiguous=None).dt.tz_convert('UTC')
    
    # Extract components using the .dt accessor
    df['Date'] = dt_utc.dt.date
    df['Year'] = dt_utc.dt.year
    df['Month'] = dt_utc.dt.month
    df['Hour'] = dt_utc.dt.hour
    
    # 3. Consolidate Aggregations
    # Instead of multiple groupbys, we build a mapping of columns to aggregate
    group_keys = ['Date', 'Year', 'Month', 'Altitude', 'Hour']
    agg_map = {
        'CO2': 'mean',
        'Latitude': 'mean'
    }
    
    # Rename mapping to handle the logic where input names differ from output names
    rename_map = {}

    if CH4_choice == 1:
        agg_map['CH4'] = 'mean'
        agg_map['CH4_STDEV'] = 'mean'
        rename_map['CH4_STDEV'] = 'CH4_STD'
        
    if CO_choice == 1:
        agg_map['CO'] = 'mean'
        
    if Temp_choice == 1:
        agg_map['Temperature'] = 'mean'
        rename_map['Temperature'] = 'Temp'

    # Single aggregation pass
    df_aircraft_agg = df.groupby(group_keys).agg(agg_map).reset_index()
    
    # Apply renames to match your exact requested output format
    if rename_map:
        df_aircraft_agg = df_aircraft_agg.rename(columns=rename_map)
        
    # 4. Final list of unique dates
    df_aircraft_date = df['Date'].unique()
    
    return df_aircraft_agg, df_aircraft_date


## GOSAT XCO2
def read_in_gosat_co2(dir_gosat, box, years):
    all_records = []
    
    # Pre-calculate box boundaries for readability
    lat_min, lat_max = box[0], box[1]
    lon_min, lon_max = box[2], box[3]

    for year in years:
        dir_gosat_year = Path(dir_gosat) / str(year)
        
        for filename in dir_gosat_year.glob('*.h5'):
            with h5py.File(filename, 'r') as f:
                # 1. Vectorized Spatial Filtering
                lats = f['Data']['geolocation']['latitude'][:]
                lons = f['Data']['geolocation']['longitude'][:]
                
                mask = (lats > lat_min) & (lats < lat_max) & \
                       (lons > lon_min) & (lons < lon_max)
                
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                # 2. File-level Constants
                date_str = os.path.basename(filename)[9:17]
                this_date = datetime.strptime(date_str, '%Y%m%d').date()
                month = this_date.month
                season = ana.get_season(this_date)

                # 3. Batch Data Extraction
                # Extract only the indices that passed the spatial mask
                xco2 = f['Data']['mixingRatio']['XCO2BiasCorrected'][indices]
                xco2_err = f['Data']['mixingRatio']['XCO2BiasCorrectedError'][indices]
                p0 = f['Data']['auxiliaryParameter']['surfacePressure'][indices]
                p0_d = f['scanAttribute']['biasCorrectionUsedData']['surfacePressureDifference'][indices]
                sa_nir = f['scanAttribute']['referenceData']['surfaceAlbedo'][indices]
                sa_swir = f['scanAttribute']['referenceData']['surfaceAlbedo'][indices]
                vza = f['Data']['geolocation']['satelliteZenith'][indices]
                sza = f['Data']['geolocation']['solarZenith'][indices]
                am = f['scanAttribute']['biasCorrectionUsedData']['airMass'][indices]
                aot = f['scanAttribute']['biasCorrectionUsedData']['aerosolOpticalThickness'][indices]
                snr = f['scanAttribute']['qualityInformation']['SNR'][indices]
                dfs = f['Data']['retrievalQuality']['CO2DFS'][indices]
                
                # Retrieval Quality arrays
                chi2 = f['Data']['retrievalQuality']['chi2'][indices]
                ak = f['Data']['retrievalQuality']['columnAveragingKernel'][indices]
                qa = f['scanAttribute']['qualityInformation']["totalPreScreeningResult"][indices]
                
                # Reference Data
                xco2_a = f['scanAttribute']['referenceData']['XCO2'][indices]
                xa_prof = f['scanAttribute']['referenceData']['CO2Profile'][indices]
                
                # 4. Pressure Weighing Function Calculation
                # Calculation: h_j = dryAirPartialColumn / dryAirTotalColumn
                partial_col = f['scanAttribute']['referenceData']['dryAirPartialColumn'][indices]
                total_col = f['Data']['auxiliaryParameter']['dryAirTotalColumn'][indices]
                # Expand total_col dimensions for element-wise division across levels
                h_j = partial_col / total_col[:, np.newaxis]

                # 5. Time Parsing (Vectorized Decoding)
                raw_times = f['scanAttribute']['time'][indices]
                # Extracting hour: slice [10:] then [1:3] from byte strings
                times = [int(t.decode("utf-8")[11:13]) for t in raw_times]

                # 6. Record Assembly
                for idx_in_mask, i in enumerate(indices):
                    this_sa_nir = sa_nir[idx_in_mask, 0]
                    if this_sa_nir <0:
                        this_sa_nir = np.nan
                    this_sa_swir = sa_swir[idx_in_mask, 2]
                    if this_sa_swir <0:
                        this_sa_swir = np.nan
                    
                    new_row = {
                        "Date": this_date, "Year": year, "Month": month, "Season": season, "time": times[idx_in_mask],
                        "lat": lats[i], "lon": lons[i],
                        "xco2_gosat": xco2[idx_in_mask], "xco2_gosat_error": xco2_err[idx_in_mask],
                        "P0": p0[idx_in_mask], "P0_d": p0_d[idx_in_mask],
                        "SA_NIR": this_sa_nir, "SA_SWIR": this_sa_swir,
                        "VZA": vza[idx_in_mask], "SZA": sza[idx_in_mask], "AM": am[idx_in_mask], "AOT": aot[idx_in_mask],
                        "SNR_CO2w": snr[idx_in_mask, 1, 2], "SNR_CO2s": snr[idx_in_mask, 3, 2], 'DOFS': dfs[idx_in_mask],
                        "Chi2": chi2[idx_in_mask], "QA": qa[idx_in_mask],
                        "xco2_a": xco2_a[idx_in_mask], "h_j": h_j[idx_in_mask],
                        "a_co2_j": ak[idx_in_mask], "x_a": xa_prof[idx_in_mask]
                    }
                    all_records.append(new_row)

    return pd.DataFrame(all_records)


## GOSAT XCH4
def read_in_gosat_ch4(dir_gosat, box, years):
    all_records = []
    
    # Pre-calculate box boundaries
    lat_min, lat_max = box[0], box[1]
    lon_min, lon_max = box[2], box[3]

    for year in years:
        dir_gosat_year = Path(dir_gosat) / str(year)
        
        for filename in dir_gosat_year.glob('*.h5'):
            with h5py.File(filename, 'r') as f:
                # 1. Spatial Masking
                lats = f['Data']['geolocation']['latitude'][:]
                lons = f['Data']['geolocation']['longitude'][:]
                
                mask = (lats > lat_min) & (lats < lat_max) & \
                       (lons > lon_min) & (lons < lon_max)
                
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                # 2. File-level Data
                date_str = os.path.basename(filename)[9:17]
                this_date = datetime.strptime(date_str, '%Y%m%d').date()
                month = this_date.month
                season = ana.get_season(this_date)

                # 3. Direct Vectorized Extractions
                xch4 = f['Data']['mixingRatio']['XCH4BiasCorrected'][indices] * 1000
                p0 = f['scanAttribute']['referenceData']['surfacePressure'][indices]
                p0_d = f['scanAttribute']['biasCorrectionUsedData']['surfacePressureDifference'][indices]
                temp_d = f['scanAttribute']['biasCorrectionUsedData']['temperatureShift'][indices]
                xh2o = f['scanAttribute']['biasCorrectionUsedData']['XH2O'][indices]
                am = f['scanAttribute']['biasCorrectionUsedData']['airMass'][indices]
                sza = f['Data']['geolocation']['solarZenith'][indices]
                vza = f['Data']['geolocation']['satelliteZenith'][indices]
                snr = f['scanAttribute']['qualityInformation']['SNR'][indices]
                dfs = f['Data']['retrievalQuality']['CH4DFS'][indices]
                
                chi2 = f['Data']['retrievalQuality']['chi2'][indices]
                qa = f['scanAttribute']['qualityInformation']["totalPreScreeningResult"][indices]
                
                xch4_a = f['scanAttribute']['referenceData']['XCH4'][indices]
                ak = f['Data']['retrievalQuality']['columnAveragingKernel'][indices]
                xa_prof = f['scanAttribute']['referenceData']['CH4Profile'][indices]

                # 4. Pressure Weighing Function Vectorization
                partial_col = f['scanAttribute']['referenceData']['dryAirPartialColumn'][indices]
                total_col = f['Data']['auxiliaryParameter']['dryAirTotalColumn'][indices]
                h_j = partial_col / total_col[:, np.newaxis]

                # 5. Aerosol Optical Thickness Vectorization
                # Extracts Fine (index 0) and Coarse (index 1) across all vertical levels
                aot_total = f['Data']['auxiliaryParameter']['aerosolOpticalThickness'][indices]
                aot = np.sum(aot_total[:, 0, :], axis=1) + np.sum(aot_total[:, 1, :], axis=1)

                # 6. Surface Albedo Vectorization
                sa_nir = f['scanAttribute']['referenceData']['surfaceAlbedo'][indices]
                sa_swir = f['scanAttribute']['referenceData']['surfaceAlbedo'][indices]

                # 7. Time Parsing
                raw_times = f['scanAttribute']['time'][indices]
                times = [int(t.decode("utf-8")[11:13]) for t in raw_times]

                # 8. Assembly
                for idx_in_mask, i in enumerate(indices):
                    this_sa_nir = sa_nir[idx_in_mask, 0]
                    if this_sa_nir <0:
                        this_sa_nir = np.nan
                    this_sa_swir = sa_swir[idx_in_mask, 2]
                    if this_sa_swir <0:
                        this_sa_swir = np.nan
                    
                    all_records.append({
                        "Date": this_date, "Year": year, "Month": month, "Season": season, "time": times[idx_in_mask],
                        "xch4_gosat": xch4[idx_in_mask],
                        "lat": lats[i], "lon": lons[i],
                        "P0": p0[idx_in_mask], "P0_d": p0_d[idx_in_mask], "Temp_d": temp_d[idx_in_mask],
                        "XH2O": xh2o[idx_in_mask], "AM": am[idx_in_mask],
                        "SA_NIR": this_sa_nir, "SA_SWIR": this_sa_swir,
                        "SZA": sza[idx_in_mask], "VZA": vza[idx_in_mask], "AOT": aot[idx_in_mask],
                        "SNR_CH4": float(snr[idx_in_mask, 2, 2]), "SNR_O2": float(snr[idx_in_mask, 0, 2]), 'DOFS': dfs[idx_in_mask],
                        "Chi2": chi2[idx_in_mask], "QA": qa[idx_in_mask],
                        "xch4_a": xch4_a[idx_in_mask], "h_j": h_j[idx_in_mask],
                        "a_ch4_j": ak[idx_in_mask], "x_a": xa_prof[idx_in_mask]
                    })

    if not all_records:
        # Return empty dataframe with the original exact column specifications
        return pd.DataFrame(columns=[
            'Date', 'Year', 'Month', 'Season', 'time', 'lat', 'lon', 'P0', 
            'xch4_gosat', 'xch4_a', 'h_j', 'a_ch4_j', 'x_a'
        ])

    return pd.DataFrame(all_records)


## CAMSinv CH4, daily
def get_camsinv_ch4(camsinv_path, lat, lon):
    all_rows = []
    
    for filename in Path(camsinv_path).glob('*.nc'):
        # Efficient filename date parsing
        date_segment = filename.name[-9:-3]
        file_dt = datetime.strptime(date_segment, '%Y%m')
        
        if file_dt.year <= 2022:
            # 1. Use Xarray's built-in selection with nearest-neighbor matching
            # This avoids converting coordinates to lists for index lookups
            ds = xr.open_dataset(filename)
            ds_loc = ds.sel(latitude=lat, longitude=lon, method='nearest')
            
            # 2. Extract arrays once to utilize NumPy broadcasting
            # Formula: P = (hyai + hybi * ps) / 100
            ps = ds_loc["ps"].values            # Shape: (time,)
            hyai = ds_loc["hyai"].values        # Shape: (level_interface,)
            hybi = ds_loc["hybi"].values        # Shape: (level_interface,)
            ch4_all = ds_loc["CH4"].values      # Shape: (time, level)
            
            # 3. Vectorized Pressure Calculations
            # Broadcast ps across the level_interface dimension
            # Resulting pres_bnd shape: (time, level_interface)
            pres_bnd = (hyai + hybi * ps[:, np.newaxis]) / 100
            
            # Mid-layer pressure: (P_i + P_{i+1}) / 2
            pres_mid = (pres_bnd[:, :-1] + pres_bnd[:, 1:]) / 2
            
            # 4. Vectorized Date Conversion
            # pd.to_datetime is significantly faster than str slicing in a loop
            t_dt = pd.to_datetime(ds_loc.time.values).date
            
            # 5. Collect data in a list (O(n) efficiency)
            for i in range(len(t_dt)):
                all_rows.append({
                    'Date': t_dt[i],
                    'Year': file_dt.year,
                    'Month': file_dt.month,
                    'Pressure': pres_mid[i].tolist(),
                    'Pressure_boundary': pres_bnd[i].tolist(),
                    'CH4': ch4_all[i].tolist()
                })
            
            ds.close()

    if not all_rows:
        return pd.DataFrame(columns=['Date', 'Year', 'Month', 'Pressure', 'Pressure_boundary', 'CH4'])

    # 6. Final construction and sorting
    out = pd.DataFrame(all_rows)
    return out.sort_values('Date', ignore_index=True)


## CAMSinv CO2, 3-hourly
def get_camsinv_co2(camsinv_path, lat, lon, target_hour):
    all_data = []
    
    # Pre-parse filenames to avoid opening files outside the date range
    for filename in Path(camsinv_path).glob('*.nc'):
        date_str = filename.name[-9:-3] # Extract YYYYMM
        d_year = int(date_str[:4])
        d_month = int(date_str[4:])
        
        if 2002 <= d_year <= 2022:
            ds = xr.open_dataset(filename)
            
            # 1. Spatial Selection (Vectorized)
            ds_loc = ds.sel(latitude=lat, longitude=lon, method='nearest')
            
            # 2. Time/Hour Filtering
            # Use Xarray's .dt accessor to get hours for all time steps at once
            file_hours = ds_loc.time.dt.hour.values
            
            # Replicating findClosestValue logic
            closest_hr_val = file_hours[np.abs(file_hours - target_hour).argmin()]
            t_indices = np.where(file_hours == closest_hr_val)[0]
            
            if len(t_indices) == 0:
                ds.close()
                continue
                
            # Subset the location-specific data to only the needed hours
            ds_filtered = ds_loc.isel(time=t_indices)
            
            # 3. Vectorized Pressure Calculations
            # P = (ap + bp * Psurf) / 100
            # Resulting shape: (time, level_interface)
            ps = ds_filtered["Psurf"].values
            ap = ds_filtered["ap"].values
            bp = ds_filtered["bp"].values
            
            # Broadcasting: (level,) + (level,) * (time, 1) -> (time, level)
            pres_bnd = (ap + bp * ps[:, np.newaxis]) / 100
            
            # Mid-layer pressure calculation using slicing (P_i + P_{i+1}) / 2
            pres_mid = (pres_bnd[:, :-1] + pres_bnd[:, 1:]) / 2
            
            # 4. Vectorized CO2 Scaling
            # Shape: (time, level)
            co2_vals = ds_filtered["CO2"].values * 1_000_000
            
            # 5. Extract dates
            t_dates = pd.to_datetime(ds_filtered.time.values).date
            
            # 6. Assembly (O(n) list collection)
            for i in range(len(t_indices)):
                all_data.append({
                    'Date': t_dates[i],
                    'Year': d_year,
                    'Month': d_month,
                    'Pressure': pres_mid[i].tolist(),
                    'Pressure_boundary': pres_bnd[i].tolist(),
                    'CO2': co2_vals[i].tolist()
                })
            
            ds.close()

    if not all_data:
        return pd.DataFrame(columns=['Date', 'Year', 'Month', 'Pressure', 'Pressure_boundary', 'CO2'])

    # Final construction and sorting
    out = pd.DataFrame(all_data)
    return out.sort_values('Date', ignore_index=True)


## SciSAT CO2, CH4 
def read_in_scisat(scisat_file_path,
                   d, target_lat, target_lon):
    
    all_records = []
    failed = []
    
    for d_val in d:
        this_d_str = str(d_val)[:7] # YYYY-MM
        try:
            # Load datasets
            path_co2 = os.path.join(scisat_file_path, 'CO2', f'ACEFTS_L2_v5p3_CO2_{this_d_str}.nc')
            path_ch4 = os.path.join(scisat_file_path, 'CH4', f'ACEFTS_L2_v5p3_CH4_{this_d_str}.nc')
            
            with xr.open_dataset(path_co2) as ds_co2, xr.open_dataset(path_ch4) as ds_ch4:
                # 1. Vectorized Spatial Search (Euclidean)
                # CO2 Search
                lons_co2 = ds_co2['longitude'].values
                lats_co2 = ds_co2['latitude'].values
                dists_co2 = np.sqrt((lons_co2 - target_lon)**2 + (lats_co2 - target_lat)**2)
                idx_co2 = np.argmin(dists_co2)
                dist_co2 = geodesic((lats_co2[idx_co2], lons_co2[idx_co2]), (target_lat, target_lon)).km
                
                # CH4 Search
                lons_ch4 = ds_ch4['longitude'].values
                lats_ch4 = ds_ch4['latitude'].values
                dists_ch4 = np.sqrt((lons_ch4 - target_lon)**2 + (lats_ch4 - target_lat)**2)
                idx_ch4 = np.argmin(dists_ch4)
                dist_ch4 = geodesic((lats_co2[idx_ch4], lons_co2[idx_ch4]), (target_lat, target_lon)).km

                # 2. Extract Data for target index
                # CO2 components
                ss_co2_all = ds_co2['CO2'].values[idx_co2]
                ss_err_co2 = ds_co2['CO2_error'].values[idx_co2]
                ss_flg_co2 = ds_co2['CO2_flag'].values[idx_co2]
                ss_alt_co2 = ds_co2['altitude'].values
                ss_pre_co2 = ds_co2['pressure'].values[idx_co2]
                
                # CH4 components
                ss_ch4_all = ds_ch4['CH4'].values[idx_ch4]
                ss_err_ch4 = ds_ch4['CH4_error'].values[idx_ch4]
                ss_alt_ch4 = ds_ch4['altitude'].values
                ss_pre_ch4 = ds_ch4['pressure'].values[idx_ch4]

                # 3. Vectorized Filtering (Masking)
                # CO2 Filter: value > 0, error > 0, flag in [2, 4]
                mask_co2 = (ss_co2_all > 0) & (ss_err_co2 > 0) & np.isin(ss_flg_co2, [2, 4])
                
                # CH4 Filter: value > 0, error > 0
                mask_ch4 = (ss_ch4_all > 0) & (ss_err_ch4 > 0)

                # 4. Record Assembly
                # Date of satellite observation
                ss_day = int(ds_co2['day'].values[idx_co2])
                ss_date_str = f"{this_d_str}-{ss_day}"
                
                if dist_co2 > 400:
                    continue
                else:
                    all_records.append({
                        'Date': d_val,
                        'Year': d_val.year,
                        'Month': d_val.month,
                        'Lat': lats_co2[idx_co2],
                        'Lon': lons_co2[idx_co2],
                        'Dist_CO2': dist_co2,
                        'Dist_CH4': dist_ch4,
                        'Date_ss': datetime.strptime(ss_date_str, '%Y-%m-%d').date(),
                        'Pressure_CO2': (ss_pre_co2[mask_co2] / 100).tolist(),
                        'Altitude_CO2': ss_alt_co2[mask_co2].tolist(),
                        'Pressure_CH4': (ss_pre_ch4[mask_ch4] / 100).tolist(),
                        'Altitude_CH4': ss_alt_ch4[mask_ch4].tolist(),
                        'CO2': (ss_co2_all[mask_co2] * 1_000_000).tolist(),
                        'CH4': (ss_ch4_all[mask_ch4] * 1_000_000_000).tolist()
                    })

        except Exception as e:
            failed.append([d_val, e])
            
    # Return as DataFrame
    if not all_records:
        return pd.DataFrame(columns=['Date', 'Year', 'Month', 'Dist']), failed
        
    return pd.DataFrame(all_records), failed


## IGRA
def read_in_igra(df_igra, h_local):
    all_records = []
    
    with open(df_igra, 'r') as igra_in:
        current_header = None
        
        for line in igra_in:
            if line.startswith('#'):
                # Header lines start with #
                # Original slicing logic equivalent: section[0][12:22]
                # Because we keep the '#', indices are shifted by +1
                year = int(line[13:17])
                
                # Pre-filter by Year and Hour to skip unnecessary record processing
                if 2002 <= year <= 2022:
                    hour = int(line[24:26])
                    if hour == h_local:
                        month = int(line[18:20])
                        day = int(line[21:23])
                        # Create date object once per header section
                        dt_obj = datetime(year, month, day).date()
                        current_header = (dt_obj, year, month, hour)
                    else:
                        current_header = None
                else:
                    current_header = None
            else:
                # Record lines
                if current_header is None:
                    continue
                
                # IGRA records use 'A' or 'B' as flags; remove them to allow int conversion
                clean_line = line.replace('B', ' ').replace('A', ' ')
                
                tp = int(clean_line[:2])
                if tp == 20:
                    continue
                
                press_raw = int(clean_line[9:15])
                if press_raw < 0:
                    continue
                    
                # unpack header context
                dt, yr, mo, hr = current_header
                
                all_records.append({
                    "Date": dt,
                    "Year": yr,
                    "Month": mo,
                    "Hour": hr,
                    "Pressure": press_raw / 100.0,
                    "Altitude": int(clean_line[16:21]),
                    "Temperature": int(clean_line[23:27]) / 10.0,
                    "Tropopause": tp
                })

    # Return empty DataFrame with correct columns if no data found
    if not all_records:
        return pd.DataFrame(columns=['Date', 'Year', 'Month', 'Hour', 'Pressure', 'Altitude', 'Temperature', 'Tropopause'])

    # Single bulk construction is much faster than repeated appending
    return pd.DataFrame(all_records)


## MERRA-2 TP, PBL, Temp
def get_merra2_tp_pbl(merra2_file_path, lat, lon, h):
    all_results = []
    failed = []
    
    nc_files = glob.glob(f'{merra2_file_path}/*.nc')
    if not nc_files:
        return pd.DataFrame(columns=['Date', 'Season', 'PBL_Pressure', 'P0', 'TP_B']), failed

    this_h = min(range(0, 24), key=lambda x: abs(x - h))
    
    # 1. Pre-calculate spatial indices using only the first file
    try:
        with xr.open_dataset(nc_files[0], engine='netcdf4') as first_ds:
            lat_i = int(np.abs(first_ds["lat"].values - lat).argmin())
            lon_i = int(np.abs(first_ds["lon"].values - lon).argmin())
    except Exception as e:
        return pd.DataFrame(), [["Initialization failed", e]]

    for f in nc_files:
        f_t = f[-15:-7]
        try:
            f_t_d = datetime.strptime(f_t, "%Y%m%d").date()
            season = ana.get_season(f_t_d)
        except Exception as e:
            failed.append([f_t, e])
            continue
            
        try:
            with xr.open_dataset(f, engine='netcdf4') as pbl_data:
                
                # Extract the exact point via index slicing before calling .values
                pbl = float(pbl_data["PBLTOP"].isel(time=this_h, lat=lat_i, lon=lon_i).values) / 100
                P0 = float(pbl_data["PS"].isel(time=this_h, lat=lat_i, lon=lon_i).values) / 100
                TP_B = float(pbl_data["TROPPB"].isel(time=this_h, lat=lat_i, lon=lon_i).values) / 100
                
                all_results.append({
                    'Date': f_t_d, 'Season': season, 'PBL_Pressure': pbl,
                    'P0': P0, 'TP_B': TP_B
                })
                # print(f_t_d)
                
        except Exception as e:
            failed.append([f_t_d, e])
 
    if not all_results:
        out = pd.DataFrame(columns=['Date', 'Season', 'PBL_Pressure', 'P0','TP_B'])
    else:
        out = pd.DataFrame(all_results)
        
    return out.sort_values(by='Date', ignore_index=True), failed

## MERRA-2 Dry air mass
def get_g_lat_height_vec(lat, hs):
    ge = 9.780327
    Re = 6371008
    lat_rad = np.radians(lat)
    # Vectorized gravity calculation
    g_lat = ge * (1 + 0.0053024 * np.sin(lat_rad)**2 - 0.0000058 * np.sin(2 * lat_rad)**2)
    return g_lat / (1 + hs / Re)**2


def get_dry_air_mass_vec(delp, g_lat_height, qv):
    return (delp / g_lat_height) * (1 - qv)


def get_MERRA2_air_mass(merra2_ad_path, this_hour, lat, lon):
    all_records = []
    failed = []
    hrs = np.arange(0, 24, 3)

    for filename in Path(merra2_ad_path).glob('*.nc4'):
        # Parse date from filename efficiently
        d_str = filename.name[-12:-4]
        d = datetime.strptime(d_str, '%Y%m%d').date()

        try:
            with xr.open_dataset(filename) as f:
                # Find closest hour index
                closest_hr = hrs[np.abs(hrs - this_hour).argmin()]
                hour_idx = np.where(hrs == closest_hr)[0][0]

                # 1. Spatial Selection using Xarray
                ds_point = f.sel(lat=lat, lon=lon, method='nearest').isel(time=hour_idx)
                    
                # 2. Extract 1D arrays (72 layers)
                delp = ds_point['DELP'].values
                qv = ds_point['QV'].values
                pl = ds_point['PL'].values
                h = ds_point['H'].values
                temp = ds_point['T'].values
                ps = ds_point['PS'].values

                # 3. Vectorized Calculations
                gs = get_g_lat_height_vec(lat, h)
                dam = get_dry_air_mass_vec(delp, gs, qv)
                ap = dam * 2.079e25

                # 4. Pressure Boundaries (using cumulative sum for performance)
                # p_top of first layer is pl[0] - delp[0]/2
                p_top_start = pl[0] - delp[0] / 2
                # The boundaries are p_top_start + cumulative sum of delp
                pressure_boundaries = np.concatenate(([p_top_start], p_top_start + np.cumsum(delp)))

                # 5. Build record (Keeping exact output units and formats)
                all_records.append({
                    'Date': d,
                    'P0': ps / 100,
                    'DELP': (delp / 100).tolist(),
                    'QV': qv.tolist(),
                    'PL': (pl / 100).tolist(),
                    'H': (h / 1000).tolist(),
                    'Temp': temp.tolist(),
                    'Pressure_boundary': (pressure_boundaries / 100).tolist(),
                    'Dry_air_mass': dam.tolist(),
                    'Dry_air_partial_column': ap.tolist()
                })
        except Exception as e:
            failed.append([d_str, str(e)])
                
    for filename in Path(merra2_ad_path).glob('*.nc'):
        # Parse date from filename efficiently
        d_str = filename.name[-15:-7]
        d = datetime.strptime(d_str, '%Y%m%d').date()
        
        try:
            with xr.open_dataset(filename) as f:
                # Find closest hour index
                closest_hr = hrs[np.abs(hrs - this_hour).argmin()]
                hour_idx = np.where(hrs == closest_hr)[0][0]

                # 1. Spatial Selection using Xarray
                ds_point = f.sel(lat=lat, lon=lon, method='nearest').isel(time=hour_idx)
                    
                # 2. Extract 1D arrays (72 layers)
                delp = ds_point['DELP'].values
                qv = ds_point['QV'].values
                pl = ds_point['PL'].values
                h = ds_point['H'].values
                temp = ds_point['T'].values
                ps = ds_point['PS'].values

                # 3. Vectorized Calculations
                gs = get_g_lat_height_vec(lat, h)
                dam = get_dry_air_mass_vec(delp, gs, qv)
                ap = dam * 2.079e25

                # 4. Pressure Boundaries (using cumulative sum for performance)
                # p_top of first layer is pl[0] - delp[0]/2
                p_top_start = pl[0] - delp[0] / 2
                # The boundaries are p_top_start + cumulative sum of delp
                pressure_boundaries = np.concatenate(([p_top_start], p_top_start + np.cumsum(delp)))

                # 5. Build record (Keeping exact output units and formats)
                all_records.append({
                        'Date': d,
                        'P0': ps / 100,
                        'DELP': (delp / 100).tolist(),
                        'QV': qv.tolist(),
                        'PL': (pl / 100).tolist(),
                        'H': (h / 1000).tolist(),
                        'Temp': temp.tolist(),
                        'Pressure_boundary': (pressure_boundaries / 100).tolist(),
                        'Dry_air_mass': dam.tolist(),
                        'Dry_air_partial_column': ap.tolist()
                        })
        except Exception as e:
            failed.append([d_str, str(e)])

    if not all_records:
        return pd.DataFrame(), failed

    # Convert to DataFrame and sort
    out_df = pd.DataFrame(all_records)
    return out_df.sort_values(by='Date', ignore_index=True), failed