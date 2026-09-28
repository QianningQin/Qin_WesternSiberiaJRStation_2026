#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for Analyses
"""
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timezone, timedelta
from datetime import date

import bisect
import math
import ccg_filter as ccgfilt
from geopy.distance import geodesic
import math
from scipy import interpolate
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from statsmodels.stats.weightstats import DescrStatsW

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

## Function to get season
Y = 2000
seasons = [(4, (date(Y,  1, 1),  date(Y, 2, 29))),
           (1, (date(Y,  3, 1),  date(Y, 5, 31))),
           (2, (date(Y,  6, 1),  date(Y, 8, 31))),
           (3, (date(Y,  9, 1),  date(Y, 11,30))),
           (4, (date(Y, 12, 1),  date(Y, 12,31)))]

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)

## Function to find closest value from list
def findClosestValue(givenList, target):
 
    def difference(givenList):
        return abs(givenList - target)
 
    result = min(givenList, key=difference)
 
    return result

## Function to find closest value on the larger side from list
def findClosestGreater(target, value_list):
    greater_values = [x for x in value_list if x >= target]

    # If no values are greater, return an indicator (e.g., None, or raise an error)
    if not greater_values:
        return None
    
    return min(greater_values)

## Add distance to GOSAT obs.
def get_distance(gosat_df, coord):
    dist = []
    for i in range(len(gosat_df)):
        dist.append(geodesic(coord, (gosat_df.lat[i], gosat_df.lon[i])).km)
    gosat_df['dist'] = dist
    return gosat_df

## Get layer-averaged pressure
def get_middle_pressure(input_list):
    if len(input_list) < 2:
        return []

    averaged_list = []
    for i in range(len(input_list) - 1):
        average = (input_list[i] + input_list[i+1]) / 2
        averaged_list.append(average)
    return averaged_list

## Generate ±1 dates to filter GOSAT obs.
def get_aircraft_gosat_dates(aircraft_date, gosat_date):
    num_days = 3
    aircraft_3d = []

    for a_d in aircraft_date:
        start_date = a_d - timedelta(days=1)
        for i in range(num_days):
            aircraft_3d.append(start_date + timedelta(days=i))
            
    filtered_gosat = set(gosat_date).intersection(set(aircraft_3d))
    
    return sorted(filtered_gosat)

## Tower afternoon mean and std
def tower_filter(tower, t_start, t_end):
    hrs = np.arange(t_start, t_end)
    failed = []
    all_results = []
    
    # Pre-filter dataset for designated hours once
    tower_hrs = tower[tower.Hour.isin(hrs)].copy()
    
    # Map each unique Date to its single MONTH value
    date_to_month = tower.groupby('Date')['MONTH'].first().to_dict()
    
    # 1. Calculate Grouped Means for GHG variables globally
    # Dropping missing values matches the .dropna().reset_index() from the original code
    ghg_means = tower_hrs.groupby(['Date', 'Hour', 'Vp'])[['CO2', 'CH4']].mean().dropna().reset_index()
    ghg_groups = ghg_means.groupby('Date')
    
    # 2. Extract maximum Pressure ('ap') across hours globally
    pressure_max_df = tower_hrs.groupby(['Date', 'Hour'])['ap'].max().dropna().reset_index()
    pressure_mapped = pressure_max_df.groupby('Date')['ap'].max().to_dict()
    
    # 3. Extract Temperature columns globally
    temp_cols = [col for col in tower.columns if col.startswith('tp')]
    if temp_cols:
        # Create a clean reference frame to calculate maximum values safely
        temp_df = tower_hrs[['Date'] + temp_cols].copy()
        for col in temp_cols:
            temp_df.loc[temp_df[col] > 50, col] = np.nan
        temp_df['max_val'] = temp_df[temp_cols].max(axis=1)
        temp_mapped = (temp_df.groupby('Date')['max_val'].max() + 273.15).to_dict()
    else:
        temp_mapped = {}
        
    ## Tower std (vertical + temporal)
    def tower_temperal_std(ghg_means_filtered, ghg_type):
        n = len(ghg_means_filtered.Vp.unique())
        N = len(ghg_means_filtered)
        
        # Handle edge case where standard formula divides by zero
        if n <= 1 or N == 0:
            return 0.0

        col = 'CO2' if ghg_type == 0 else 'CH4'
        
        # Vectorized calculation of sum of squared differences from the group means
        means = ghg_means_filtered.groupby('Vp')[col].transform('mean')
        sq_diff_sum = ((ghg_means_filtered[col] - means) ** 2).sum()
        
        v = np.sqrt(sq_diff_sum / (n - 1)) / N
        return v
    
    for d in tower.Date.unique():
        if d not in ghg_groups.groups:
            continue
        try:
            test_month = date_to_month.get(d, np.nan)
            sub_ghg = ghg_groups.get_group(d)
            
            co2_std_t = tower_temperal_std(sub_ghg, 0)
            co2_std_v = sub_ghg.groupby('Hour').CO2.mean().std()
            co2_std = np.sqrt(co2_std_v ** 2 + co2_std_t ** 2)
            
            # ch4_std_v calculates type 1 (CH4) via the temporal helper
            # ch4_std_t calculates standard variation across Hours
            ch4_std_v = tower_temperal_std(sub_ghg, 1)
            ch4_std_t = sub_ghg.groupby('Hour').CH4.mean().std()
            ch4_std = np.sqrt(ch4_std_v ** 2 + ch4_std_t ** 2)
            
            # Fetch globally mapped pressure and temperature values
            pressure_filtered = pressure_mapped.get(d, np.nan)
            temp_filtered = temp_mapped.get(d, np.nan)
            
            all_results.append({
                'Date': d, 'Month': test_month, 
                'CH4': sub_ghg.CH4.mean(), 'CO2': sub_ghg.CO2.mean(),
                'CH4_std': ch4_std, 'CH4_std_v': ch4_std_v, 'CH4_std_t': ch4_std_t,
                'CO2_std': co2_std, 'CO2_std_v': co2_std_v, 'CO2_std_t': co2_std_t,
                'Pressure': pressure_filtered, 'Temp': temp_filtered
            })
                
        except Exception as e:
            failed.append([d, e])
    
    # 4. Construct output DataFrame in a single pass
    if not all_results:
        out = pd.DataFrame(columns=['Date', 'Month', 'CH4', 'CO2', 'CH4_std', 'CH4_std_v', 
                                    'CH4_std_t', 'CO2_std', 'CO2_std_v', 'CO2_std_t', 'Pressure', 'Temp'])
    else:
        out = pd.DataFrame(all_results)
        
    out = out.sort_values('Date', ignore_index=True)
    return out, failed

## IGRA Barometric curve fitting
def curve_fitting(igra):
    # 1. Pre-calculate Monthly P0 backup (Months 1-12)
    p0_monthly = igra.loc[igra.Tropopause.isin([11, 21])].groupby('Month')['Pressure'].mean()
    p0_months_avail = p0_monthly.index.values.astype(int)

    # 2. Pre-calculate Date-specific Tropopause info to avoid .loc in the loop
    # P0 candidates (21, 11) and Tropopause_P (22)
    p0_candidates = igra.loc[igra.Tropopause.isin([11, 21])].groupby('Date')['Pressure'].max()
    tp_22_pressures = igra.loc[igra.Tropopause == 22].groupby('Date')['Pressure'].max()

    # 3. Prepare data for fitting
    igra_tofit = igra[igra.Altitude > 0].copy()
    all_results = []

    # 4. Group by Date for vectorized iteration
    for d, group in igra_tofit.groupby('Date'):
        x = group['Altitude'].values
        y = group['Pressure'].values
        
        if len(x) <= 2:
            continue

        # Resolve P0
        p0 = p0_candidates.get(d, np.nan)
        if np.isnan(p0):
            # Fallback to monthly mean using findClosestValue logic
            p0 = p0_monthly[findClosestValue(p0_months_avail, d.month)]

        # Resolve Tropopause_P
        tp_p = tp_22_pressures.get(d, np.nan)

        # 5. Curve Fitting
        # Using a lambda to pass p0 into the function scope without global reliance
        def Barometric_model(x_val, a, b, H):
            return p0 * np.exp(-a * (x_val / H) + b)

        try:
            # Initial guess [a, b, H] for better convergence
            popt, _ = curve_fit(Barometric_model, x, y, p0=[1, 1, 7000])
            a_fit, b_fit, h_fit = popt
            
            ans = p0 * np.exp(-a_fit * (x / h_fit) + b_fit)
            
            # Calculate Tropopause Altitude if Pressure exists
            tp_h = np.nan
            if not np.isnan(tp_p):
                tp_h = -h_fit / a_fit * (np.log(tp_p / p0) - b_fit)

            # Calculate RMSE
            rmse = np.sqrt(np.mean((y - ans)**2))

            all_results.append({'Date': d, 'Year': d.year, 'Month': d.month,
                                'Season': get_season(d),
                                'a': a_fit, 'b': b_fit, 'H': h_fit, 'P0': p0,
                                'Altitude': x, 'Pressure': ans,
                                'Tropopause_P': tp_p, 'Tropopause_H': tp_h,
                                'y': y, 'rmse': rmse})
        except (RuntimeError, ValueError):
            # Skip if fit fails to converge
            continue

    # 6. Bulk construction (O(n))
    return pd.DataFrame(all_results)

# Get param from the same date or the closest Year Month
def get_fitted(d, aircraft_height, param, key):
    
    if key == 1:
        this_param = param.loc[param.Date == d].reset_index().iloc[0]
    else:
        y = d.year
        m = d.month
        close_y = findClosestValue(param.loc[param.Month == m].Year.unique(), y)
        this_param = param.loc[(param.Year == close_y) & (param.Month == m)].reset_index().iloc[0]
    
    p0 = this_param.P0
    a = this_param.a
    b = this_param.b
    H = this_param.H
    
    ## Aircraft height to pressure
    def Barometric(x_val, a, b, H):
        return p0 * np.exp(-a * (x_val / H) + b)
    ans = Barometric(aircraft_height, a, b, H)
   
    return ans, p0, a, b, H

def get_nan_mask(arr):
    return ~np.isnan(arr)

def make_more_column(aircraft, tower, igra_param, tp_merra2, gosat_date, gas_choice):
    all_results = []
    failed = []
    
    # 1. Pre-index data
    tp_lookup = tp_merra2.set_index('Date').to_dict('index')
    tower_lookup = tower.set_index('Date').to_dict('index')
    
    aircraft_dates = aircraft.Date.unique()
    igra_param_dates = set(igra_param.Date.unique())
    d_tower_set = set(tower.Date.unique())

    # Map gas choices to column names
    gas_map = {1: ('CH4', 'CH4_column', 'CH4_tower_std', 'CH4_std'), 
               0: ('CO2', 'CO2_column', 'CO2_tower_std', 'CO2_std'), 
               2: ('CO', 'CO_column', None, None)}
    
    gas_key, gas_col, std_col, tower_std_key = gas_map.get(gas_choice, (None, None, None, None))

    # 2. Single loop over all dates
    for d in gosat_date:
        key = 1 if d in igra_param_dates else 0
        try:
            # Metadata
            y, m, s = d.year, d.month, get_season(d)
            
            # --- Surface Pressure Logic (Priority: Tower > Model) ---
            tp_info = tp_lookup.get(d)
            if tp_info is None:
                continue
            model_p0 = tp_info['P0']
            
            final_p0 = model_p0
            if d in d_tower_set:
                t_info = tower_lookup[d]
                t_pres = t_info.get('Pressure', np.nan)
                if not np.isnan(t_pres) and t_pres > 10:
                    final_p0 = t_pres
            
            # --- Aircraft Data Processing ---
            aircraft_d = findClosestValue(aircraft_dates, d)
            this_aircraft = aircraft[aircraft.Date == aircraft_d]
            
            # Vectorized cleaning: Remove NaNs AND 0s from aircraft gas data
            raw_height = this_aircraft.Altitude.values
            raw_temp = this_aircraft.Temp.values + 273.15
            raw_gas = this_aircraft[gas_key].values
            
            mask = (raw_gas > 0) & (~np.isnan(raw_gas))
            aircraft_height = raw_height[mask]
            aircraft_temp = raw_temp[mask]
            aircraft_gas = raw_gas[mask]
            aircraft_hour = this_aircraft.Hour.min()
            
            # Pressure Fitting
            aircraft_pressure, this_P0, this_a, this_b, this_H = get_fitted(d, aircraft_height, igra_param, key)
            
            # --- Tower Data Processing (Fallback to Lowest Aircraft) ---
            tower_val, tower_std = np.nan, np.nan
            tower_temp = np.nan
            
            # A. Attempt to get gas and temp from tower dataset
            if d in d_tower_set:
                t_info = tower_lookup[d]
                tower_temp = t_info['Temp']
                if gas_choice != 2:
                    raw_val = t_info[gas_key]
                    if not np.isnan(raw_val) and raw_val > 0:
                        tower_val = raw_val
                        tower_std = t_info[tower_std_key]

            # B. Fallback for Gas: If tower is missing/0, use aircraft data at minimum altitude
            if gas_choice != 2 and (np.isnan(tower_val) or tower_val <= 0):
                if aircraft_gas.size > 0:
                    idx_lowest = np.argmin(aircraft_height)
                    tower_val = aircraft_gas[idx_lowest]
                    tower_std = np.nan # Standard deviation unknown for aircraft fallback
            
            # C. Fallback for Temp: Use aircraft at minimum altitude if tower temp is missing
            if (np.isnan(tower_temp) or tower_temp <= 0) and aircraft_temp.size > 0:
                # Use height mask to find the temperature closest to the ground
                idx_lowest_t = np.argmin(aircraft_height)
                tower_temp = aircraft_temp[idx_lowest_t]

            # 3. Build Result Dictionary (exact column mapping)
            res = {
                'Date': d, 'Year': y, 'Month': m, 'Hour': aircraft_hour, 'Season': s,
                'a': this_a, 'b': this_b, 'H': this_H, 'P0': final_p0
            }
            
            # Handle list/array concatenations
            if gas_choice in [0, 1]:
                res['Altitude'] = np.append(0, aircraft_height)
                res['Pressure'] = np.append(final_p0, aircraft_pressure)
                res['Temp'] = np.append(tower_temp, aircraft_temp)
                res[gas_col] = np.append(tower_val, aircraft_gas)
                res[std_col] = tower_std
            else: # gas_choice == 2 (CO)
                res['Altitude'] = aircraft_height
                res['Pressure'] = aircraft_pressure
                res['Temp'] = aircraft_temp
                res[gas_col] = aircraft_gas
            
            all_results.append(res)
            
        except Exception as e:
            failed.append([d, e])

    # 4. Final Construction
    if not all_results:
        return pd.DataFrame(), failed
        
    fitted = pd.DataFrame(all_results).sort_values('Date').reset_index(drop=True)
    return fitted, failed

## Huge function to intra/extrapolate CH4 from surface to TP and above
def fit_constant_gas(p_start, p_end, gas_val, step):
    if step <= 0: 
        return np.array([]), np.array([])
    X = np.linspace(p_start, p_end, num=step, endpoint=True)
    
    # FIX: Change 0.0 to np.nan so missing data doesn't create 0 ppm/ppb layers
    val = gas_val if not np.isnan(gas_val) else np.nan
    
    Y = np.full(step, val)
    return X, Y

def fit_interp_safe(pressure, gas, step, p_range):
    if step <= 0: 
        return np.array([]), np.array([])
    
    # 1. Filter out NaNs
    mask = ~np.isnan(gas) & ~np.isnan(pressure)
    p_clean = pressure[mask]
    g_clean = gas[mask]

    # 2. Handle cases with insufficient data
    if p_clean.size < 2:
        # If only 1 point exists, we can only return a constant value
        val = g_clean[0] if p_clean.size == 1 else np.nan
        X = np.linspace(p_range[0], p_range[1], num=step, endpoint=True)
        return X, np.full(step, val)

    # 3. Handle duplicate pressures (The cause of the RuntimeWarning)
    # If multiple gas values exist for the same pressure, we take the mean
    if np.any(np.diff(np.sort(p_clean)) == 0):
        # Using a temporary pandas Series is the fastest way to group and mean duplicates
        temp = pd.Series(g_clean, index=p_clean).groupby(level=0).mean()
        p_clean = temp.index.values
        g_clean = temp.values

    # 4. Re-check size after removing duplicates
    if p_clean.size < 2:
        val = g_clean[0] if p_clean.size == 1 else np.nan
        X = np.linspace(p_range[0], p_range[1], num=step, endpoint=True)
        return X, np.full(step, val)

    # 5. Perform Interpolation
    f = interpolate.interp1d(p_clean, g_clean, bounds_error=False, fill_value="extrapolate")
    X = np.linspace(p_range[0], p_range[1], num=step, endpoint=True)
    
    # Check if p_range itself contains NaNs (e.g. if p0 or p_tp is NaN)
    if np.any(np.isnan(X)):
        return X, np.full(step, np.nan)
        
    return X, f(X)

def CH4_column_generator(column, camsinv):
    all_results = []
    failed = []
    
    # Pre-align datasets
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 1. Variables and Scaling
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            ch4_air = np.array(row.CH4_column[1:]) - 5
            ch4_tower = row.CH4_column[0] - 5
            ch4_tower_std = row.CH4_tower_std
            ch4_cams = np.array(row.CH4_cams)
            
            if len(row.CH4_column) < 5:
                failed.append([d, 'not enough obs'])
                continue

            # 2. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### TOP LEVEL: 0.5 to 0.1 hPa ###
            n_na = max(0, math.ceil(n_all * (min(p_cams) - model_top) / p_range_total))
            p_col_0, ch4_col_0 = fit_constant_gas(min(p_cams), model_top, ch4_cams[-1], n_na)

            ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
            # Logic: We use tower data and any aircraft data that falls below/within the PBL
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                # Use numpy masks instead of bisect for descending pressure arrays
                mask_in_pbl = p_air >= p_pbl
                p_pbl_points = p_air[mask_in_pbl]
                ch4_pbl_points = ch4_air[mask_in_pbl]
                
                # Combine Tower + Aircraft points inside PBL for the spline
                p_for_spline = np.append(p0, p_pbl_points)
                ch4_for_spline = np.append(ch4_tower, ch4_pbl_points)
                
                # The "edge" is the highest aircraft measurement (lowest pressure) inside the PBL
                p_edge = p_pbl_points[-1] if p_pbl_points.size > 0 else p0
                ch4_edge = ch4_pbl_points[-1] if ch4_pbl_points.size > 0 else ch4_tower
                
                # Segment 2a: Spline from P0 to the aircraft edge
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, ch4_col_2a = fit_interp_safe(p_for_spline, ch4_for_spline, n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill from aircraft edge to PBL top
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, ch4_col_2b = fit_constant_gas(p_edge, p_pbl, ch4_edge, n_2b)
                
                p_col_2 = np.append(p_col_2a[::-1], p_col_2b)
                ch4_col_2 = np.append(ch4_col_2a[::-1], ch4_col_2b)
                
                # Sort upward (Surface to PBL Top)
                s_idx = np.argsort(p_col_2)
                p_col_2, ch4_col_2 = p_col_2[s_idx][::-1], ch4_col_2[s_idx][::-1]
            else:
                # If no aircraft in PBL, just fit P0 to PBL top
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                fill_val = ch4_air[0] if np.isnan(ch4_tower) else ch4_tower
                p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, fill_val, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
            # 1a: TP to CAMSinv top
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, ch4_col_1a = fit_interp_safe(p_cams[idx_01a:], ch4_cams[idx_01a:], n_1a, [p_tp, min(p_cams)])

            # 1b: Aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_air, ch4_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c2)
                p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                ch4_above = ch4_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_above, ch4_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, p_start_c, ch4_above[0], n_1c2)
                    p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                else:
                    p_col_1c, ch4_col_1c = np.array([]), np.array([])

            p_col_1 = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            ch4_col_1 = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
            s_idx_1 = np.argsort(p_col_1)
            p_col_1, ch4_col_1 = p_col_1[s_idx_1], ch4_col_1[s_idx_1]

            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_0': ch4_col_0,
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_0': p_col_0,
                'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                'Pressure_camsinv': p_cams, 'CH4_camsinv': ch4_cams,
                'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': ch4_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    return pd.DataFrame(all_results), failed

## Huge function to intra/extrapolate CO2 from surface to TP and above
def CO2_column_generator(column, camsinv):
    all_results = []
    failed = []
    
    # 1. Pre-align datasets to eliminate search overhead in the loop
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 2. Extract and sanitize variables
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            co2_air = np.array(row.CO2_column[1:])
            co2_tower = row.CO2_column[0]
            co2_cams = np.array(row.CO2_cams)
            
            if len(row.CO2_column) < 5:
                failed.append([d, 'not enough obs'])
                continue

            # 3. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                mask_in_pbl = p_air >= p_pbl
                p_pbl_pts = p_air[mask_in_pbl]
                co2_pbl_pts = co2_air[mask_in_pbl]
                
                p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                co2_edge = co2_pbl_pts[-1] if co2_pbl_pts.size > 0 else co2_tower
                
                # Segment 2a: Spline
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, co2_col_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                                       np.append(co2_tower, co2_pbl_pts), 
                                                       n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, co2_col_2b = fit_constant_gas(p_edge, p_pbl, co2_edge, n_2b)
                
                # GLOBAL SORT UPWARD (Descending Pressure)
                p_col_2_unsorted = np.concatenate([p_col_2a, p_col_2b])
                co2_col_2_unsorted = np.concatenate([co2_col_2a, co2_col_2b])
                
                s_idx_2 = np.argsort(p_col_2_unsorted)[::-1] # High to Low
                p_col_2 = p_col_2_unsorted[s_idx_2]
                co2_col_2 = co2_col_2_unsorted[s_idx_2]
            else:
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
            # 1a: TP to model top (CAMSinv)
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, co2_col_1a = fit_interp_safe(p_cams[idx_01a:], co2_cams[idx_01a:], n_1a, [p_tp, model_top])

            # 1b: Highest aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                # Interpolate across the aircraft range
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, co2_col_1c1 = fit_interp_safe(p_air, co2_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                # Fill PBL to lowest aircraft
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c2)
                p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                co2_above = co2_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, co2_col_1c1 = fit_interp_safe(p_above, co2_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, p_start_c, co2_above[0], n_1c2)
                    p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                else:
                    p_col_1c, co2_col_1c = np.array([]), np.array([])

            # GLOBAL SORT UPWARD (Descending Pressure)
            p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
            
            s_idx_1 = np.argsort(p_col_1_unsorted)[::-1] # High to Low
            p_col_1 = p_col_1_unsorted[s_idx_1]
            co2_col_1 = co2_col_1_unsorted[s_idx_1]
            
            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1,
                'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                'Pressure_camsinv': p_cams, 'CO2_camsinv': co2_cams,
                'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results), failed

## Huge function to intra/extrapolate CO from PBL to TP
def CO_PBL_TP_generator(column):
    all_results = []
    failed = []
    
    # Process unique dates from the aircraft/boundary layer dataframe
    ds_column = column.Date.unique()

    for d in ds_column:
        try:
            # Extract row data
            row = column[column.Date == d].iloc[0]
            
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p0 = row.P0
            
            # Extract CO and Pressure columns (assuming CO is stored in 'CO_column')
            # Based on make_more_column logic, aircraft starts at index 1
            p_air = np.array(row.Pressure[1:])
            co_air = np.array(row.CO_column[1:])
            
            # 1. Clean Data: Remove NaNs and Zeros
            mask = (co_air > 0) & (~np.isnan(co_air))
            p_air = p_air[mask]
            co_air = co_air[mask]
            
            if len(p_air) < 2:
                failed.append([d, 'insufficient valid CO observations'])
                continue

            # 2. Grid Setup (0.01 hPa resolution)
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### SEGMENT 1: Highest Aircraft to Tropopause ###
            # Constant extrapolation from top-most flight measurement to TP
            n_tp_extrap = max(0, math.ceil(n_all * (np.nanmin(p_air) - p_tp) / p_range_total))
            p_seg_tp, co_seg_tp = fit_constant_gas(np.nanmin(p_air), p_tp, co_air[-1], n_tp_extrap)

            ### SEGMENT 2: PBL to Highest Aircraft ###
            # If the aircraft flew below the PBL, we spline the section within the PBL range
            if np.nanmax(p_air) > p_pbl:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                co_above = co_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_node = p_above[0]
                    # Spline through aircraft above PBL
                    n_spline = max(0, math.ceil(n_all * (p_start_node - np.nanmin(p_air)) / p_range_total))
                    p_seg_spline, co_seg_spline = fit_interp_safe(p_above, co_above, n_spline, [p_start_node, np.nanmin(p_air)])
                    
                    # Extrapolate from PBL up to the first aircraft node
                    n_pbl_extrap = max(0, math.ceil(n_all * (p_pbl - p_start_node) / p_range_total))
                    p_seg_pbl, co_seg_pbl = fit_constant_gas(p_pbl, p_start_node, co_above[0], n_pbl_extrap)
                    
                    p_mid_unsorted = np.concatenate([p_seg_pbl, p_seg_spline])
                    co_mid_unsorted = np.concatenate([co_seg_pbl, co_seg_spline])
                else:
                    p_mid_unsorted, co_mid_unsorted = np.array([]), np.array([])
            
            # If the PBL is below all aircraft measurements
            else:
                # Spline through the entire flight range
                n_spline = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_seg_spline, co_seg_spline = fit_interp_safe(p_air, co_air, n_spline, [np.nanmax(p_air), np.nanmin(p_air)])
                
                # Extrapolate from PBL up to the lowest aircraft measurement
                n_pbl_extrap = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_seg_pbl, co_seg_pbl = fit_constant_gas(p_pbl, np.nanmax(p_air), co_air[0], n_pbl_extrap)
                
                p_mid_unsorted = np.concatenate([p_seg_pbl, p_seg_spline])
                co_mid_unsorted = np.concatenate([co_seg_pbl, co_seg_spline])

            # 3. Final Assembly and Upward Sorting
            p_final = np.concatenate([p_mid_unsorted, p_seg_tp])
            co_final = np.concatenate([co_mid_unsorted, co_seg_tp])
            
            # Sort Upward (Surface to Top = Descending Pressure)
            s_idx = np.argsort(p_final)[::-1]
            
            all_results.append({
                'Date': d,
                'Year': row.Year,
                'Season': row.Season,
                'Hour': row.Hour,
                'Pressure_PBL': p_pbl,
                'Pressure_TP': p_tp,
                'CO_column': co_final[s_idx],
                'Pressure_column': p_final[s_idx],
                'Pressure_aircraft': p_air,
                'CO_aircraft': co_air,
                'P0': p0
            })

        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results).sort_values('Date').reset_index(drop=True), failed

## MERRA-2 calculation
def cal_merra2_ghg(merra2, column, ghg_choice):
    fail = []
    all_rows = []
    
    # 1. Pre-index data for O(1) dictionary lookups
    # This avoids scanning the entire DataFrame in every iteration
    col_map = column.set_index('Date').to_dict('index')
    # Pre-clean MERRA-2 and map by Date
    merra2_cleaned = merra2.drop_duplicates(subset=['Date'])
    m2_map = merra2_cleaned.set_index('Date').to_dict('index')
    
    unique_dates = merra2.Date.unique()

    for d in unique_dates:
        try:
            # Verify date exists in tower/aircraft column
            col_match = col_map.get(d)
            m2_row = m2_map.get(d)
            
            if col_match is None or m2_row is None:
                continue

            # 2. Combine observations based on ghg_choice
            # 0 = CO2, Else = CH4
            if ghg_choice == 0:
                target_p = np.concatenate([np.atleast_1d(col_match['Pressure_2']), 
                                           np.atleast_1d(col_match['Pressure_1'])])
                target_ghg = np.concatenate([np.atleast_1d(col_match['CO2_2']), 
                                             np.atleast_1d(col_match['CO2_1'])])
                target_ghg_std = col_match['CO2_tower_std']
            else:                    
                target_p = np.concatenate([np.atleast_1d(col_match['Pressure_2']), 
                                           np.atleast_1d(col_match['Pressure_1']), 
                                           np.atleast_1d(col_match['Pressure_0'])])
                target_ghg = np.concatenate([np.atleast_1d(col_match['CH4_2']), 
                                             np.atleast_1d(col_match['CH4_1']), 
                                             np.atleast_1d(col_match['CH4_0'])])
                target_ghg_std = col_match['CH4_tower_std']

            # 3. Get MERRA-2 profile
            total_p = m2_row['Pressure_boundary']
            partial_column = m2_row['Dry_air_partial_column']
            ps = m2_row.get('P0') # Surface pressure

            # 4. Clipping
            # Find closest index to 0.1 hPa using vectorized search
            total_p_arr = np.array(total_p)
            end_idx = np.abs(total_p_arr - 0.1).argmin()
            
            clip_total_p = total_p_arr[end_idx:]
            clip_partial_column = np.array(partial_column)[end_idx:]

            # 5. Binning
            binned_means, x, _ = binned_statistic(target_p, target_ghg, statistic='mean', bins=clip_total_p)
            
            # 6. Vectorized Fill NaNs (Immediate neighbors average)
            s = pd.Series(binned_means)
            neighbor_avg = pd.concat([s.shift(1), s.shift(-1)], axis=1).mean(axis=1)
            binned_means_filled = np.where(np.isnan(binned_means), neighbor_avg, binned_means)
            
            # 7. Weighted Average Calculation
            nan_mask = np.isnan(binned_means_filled)
            clean_ghg = binned_means_filled[~nan_mask]
            clean_weights = clip_partial_column[~nan_mask]
            
            if len(clean_weights) == 0:
                raise ValueError("All bins are NaN for this profile")

            merra2_ghg = np.sum(clean_ghg * clean_weights) / np.sum(clean_weights)
            
            # 8. Assembly
            bins_center = (x[:-1] + x[1:]) / 2
            
            all_rows.append({
                'Date': d, 'Year': d.year, 'Month': d.month, 'Season': get_season(d),
                'GHG_merra2': merra2_ghg, 
                'Pressure_column': target_p, 'GHG_column': target_ghg, 'GHG_tower_std': target_ghg_std,
                'Boundary': x, 'bins': bins_center, 'binned_ghg': binned_means, 
                'dry_air_partial_column': clip_partial_column
            })

        except Exception as e:
            fail.append([d, str(e)])
    
    # 9. Batch construction (much faster than repeated _append)
    if not all_rows:
        return pd.DataFrame(), fail
        
    out = pd.DataFrame(all_rows)
    return out.sort_values('Date').reset_index(drop=True), fail

## Tropopause perturbation analysis (CH4)
def get_calibrated_tp(igra, merra2):
    # Drop rows missing Tropopause_P before the merge
    igra_clean = igra.dropna(subset=['Tropopause_P'])

    # Left merge retains all merra2 dates; introduces NaNs for missing igra data
    df = pd.merge(merra2, igra_clean[['Date', 'Tropopause_P']], on='Date', how='left')
    
    regs_list = []
    
    for s in [1, 2, 3, 4]:
        this_df = df[df['Season'] == s]

        valid_data = this_df.dropna(subset=['TP_B', 'Tropopause_P'])
        
        if not valid_data.empty:
            # Reshape implicitly handled by selecting as DataFrame
            X = valid_data[['TP_B']] 
            y = valid_data['Tropopause_P']
            
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r_squared = reg.score(X, y)
            
            # Vectorized RMSE calculation
            predictions = reg.predict(X)
            rmse = np.sqrt(np.mean((y - predictions)**2))
        else:
            # Handle cases where a season lacks overlapping data
            slope, intercept, r_squared, rmse = np.nan, np.nan, np.nan, np.nan

        regs_list.append({
            'slope': slope, 
            'int': intercept, 
            'R2': r_squared, 
            'RMSE': rmse, 
            'Season': s
        })

    # Construct the regression summary DataFrame once
    regs = pd.DataFrame(regs_list)
    
    # Vectorized construction of the final output DataFrame
    out = df[['Date', 'Season', 'TP_B', 'Tropopause_P']].copy()
    
    out = out.rename(columns={
        'TP_B': 'TP_merra2',
        'Tropopause_P': 'TP_igra'
    })

    # Enforce standard column ordering and sort by date
    out = out[['Date', 'Season', 'TP_merra2', 'TP_igra']]
    out = out.sort_values(by='Date', ignore_index=True)
    
    return out, regs

def CH4_column_aj(column, cams, direction):
    all_results = []
    failed = []
    
    # Pre-align datasets
    combined = pd.merge(
        column, 
        cams[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 1. Variables and Scaling
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            ch4_air = np.array(row.CH4_column[1:]) - 5
            ch4_tower = row.CH4_column[0] - 5
            ch4_tower_std = row.CH4_tower_std
            ch4_cams = np.array(row.CH4_cams)

            # Aircraft perturbation part
            if direction == 1:
                ch4_air += 1.7
            else:
                ch4_air -= 1.7
            
            if len(row.CH4_column) < 5:
                failed.append([d, 'not enough obs'])
                continue

            # 2. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### TOP LEVEL: 0.5 to 0.1 hPa ###
            n_na = max(0, math.ceil(n_all * (min(p_cams) - model_top) / p_range_total))
            p_col_0, ch4_col_0 = fit_constant_gas(min(p_cams), model_top, ch4_cams[-1], n_na)

            ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
            # Logic: use tower data and any aircraft data that falls below/within the PBL
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                # Use numpy masks instead of bisect for descending pressure arrays
                mask_in_pbl = p_air >= p_pbl
                p_pbl_points = p_air[mask_in_pbl]
                ch4_pbl_points = ch4_air[mask_in_pbl]
                
                # Combine Tower + Aircraft points inside PBL for the spline
                p_for_spline = np.append(p0, p_pbl_points)
                ch4_for_spline = np.append(ch4_tower, ch4_pbl_points)
                
                # The "edge" is the highest aircraft measurement (lowest pressure) inside the PBL
                p_edge = p_pbl_points[-1] if p_pbl_points.size > 0 else p0
                ch4_edge = ch4_pbl_points[-1] if ch4_pbl_points.size > 0 else ch4_tower
                
                # Segment 2a: Spline from P0 to the aircraft edge
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, ch4_col_2a = fit_interp_safe(p_for_spline, ch4_for_spline, n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill from aircraft edge to PBL top
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, ch4_col_2b = fit_constant_gas(p_edge, p_pbl, ch4_edge, n_2b)
                
                p_col_2 = np.append(p_col_2a[::-1], p_col_2b)
                ch4_col_2 = np.append(ch4_col_2a[::-1], ch4_col_2b)
                
                # Sort upward (Surface to PBL Top)
                s_idx = np.argsort(p_col_2)
                p_col_2, ch4_col_2 = p_col_2[s_idx][::-1], ch4_col_2[s_idx][::-1]
            else:
                # If no aircraft in PBL, just fit P0 to PBL top
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                fill_val = ch4_air[0] if np.isnan(ch4_tower) else ch4_tower
                p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, fill_val, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
            # 1a: TP to CAMSinv top
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, ch4_col_1a = fit_interp_safe(p_cams[idx_01a:], ch4_cams[idx_01a:], n_1a, [p_tp, min(p_cams)])

            # 1b: Aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_air, ch4_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c2)
                p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                ch4_above = ch4_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_above, ch4_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, p_start_c, ch4_above[0], n_1c2)
                    p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                else:
                    p_col_1c, ch4_col_1c = np.array([]), np.array([])

            p_col_1 = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            ch4_col_1 = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
            s_idx_1 = np.argsort(p_col_1)
            p_col_1, ch4_col_1 = p_col_1[s_idx_1], ch4_col_1[s_idx_1]

            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_0': ch4_col_0,
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_0': p_col_0,
                'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                'Pressure_camsinv': p_cams, 'CH4_camsinv': ch4_cams,
                'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': ch4_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    return pd.DataFrame(all_results), failed

def CO2_column_aj(column, cams, direction):
    all_results = []
    failed = []
    
    # 1. Pre-align datasets to eliminate search overhead in the loop
    combined = pd.merge(
        column, 
        cams[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 2. Extract and sanitize variables
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            co2_air = np.array(row.CO2_column[1:])
            co2_tower = row.CO2_column[0]
            co2_cams = np.array(row.CO2_cams)
            
            if len(row.CO2_column) < 5:
                failed.append([d, 'not enough obs'])
                continue
            
            # Aircraft adjustment part
            if direction == 1:
                co2_air += .3
            else:
                co2_air -= .3
            # 3. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                mask_in_pbl = p_air >= p_pbl
                p_pbl_pts = p_air[mask_in_pbl]
                co2_pbl_pts = co2_air[mask_in_pbl]
                
                p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                co2_edge = co2_pbl_pts[-1] if co2_pbl_pts.size > 0 else co2_tower
                
                # Segment 2a: Spline
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, co2_col_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                                       np.append(co2_tower, co2_pbl_pts), 
                                                       n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, co2_col_2b = fit_constant_gas(p_edge, p_pbl, co2_edge, n_2b)
                
                # GLOBAL SORT UPWARD (Descending Pressure)
                p_col_2_unsorted = np.concatenate([p_col_2a, p_col_2b])
                co2_col_2_unsorted = np.concatenate([co2_col_2a, co2_col_2b])
                
                s_idx_2 = np.argsort(p_col_2_unsorted)[::-1] # High to Low
                p_col_2 = p_col_2_unsorted[s_idx_2]
                co2_col_2 = co2_col_2_unsorted[s_idx_2]
            else:
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
            # 1a: TP to model top (CAMSinv)
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, co2_col_1a = fit_interp_safe(p_cams[idx_01a:], co2_cams[idx_01a:], n_1a, [p_tp, model_top])

            # 1b: Highest aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                # Interpolate across the aircraft range
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, co2_col_1c1 = fit_interp_safe(p_air, co2_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                # Fill PBL to lowest aircraft
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c2)
                p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                co2_above = co2_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, co2_col_1c1 = fit_interp_safe(p_above, co2_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, p_start_c, co2_above[0], n_1c2)
                    p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                else:
                    p_col_1c, co2_col_1c = np.array([]), np.array([])

            # GLOBAL SORT UPWARD (Descending Pressure)
            p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
            
            s_idx_1 = np.argsort(p_col_1_unsorted)[::-1] # High to Low
            p_col_1 = p_col_1_unsorted[s_idx_1]
            co2_col_1 = co2_col_1_unsorted[s_idx_1]
            
            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1,
                'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                'Pressure_camsinv': p_cams, 'CO2_camsinv': co2_cams,
                'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results), failed

def get_tp_seasonal_var(tp_reg):
    return tp_reg.set_index('Season').reindex([1, 2, 3, 4]).RMSE.tolist()
    
def CH4_column_tp_adj(column, camsinv, tp_rmse, direction):
    all_results = []
    failed = []
    
    # Pre-align datasets
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 1. Variables and Scaling
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            ch4_air = np.array(row.CH4_column[1:]) - 5
            ch4_tower = row.CH4_column[0] - 5
            ch4_tower_std = row.CH4_tower_std
            ch4_cams = np.array(row.CH4_cams)

            # TP perturbation part
            this_tp_rmse = tp_rmse[row.Season-1]
            if direction == 1:
                p_tp += this_tp_rmse
            else:
                p_tp -= this_tp_rmse
            
            if len(row.CH4_column) < 5:
                failed.append([d, 'not enough obs'])
                continue

            # 2. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### TOP LEVEL: 0.5 to 0.1 hPa ###
            n_na = max(0, math.ceil(n_all * (min(p_cams) - model_top) / p_range_total))
            p_col_0, ch4_col_0 = fit_constant_gas(min(p_cams), model_top, ch4_cams[-1], n_na)

            ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
            # Logic: use tower data and any aircraft data that falls below/within the PBL
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                # Use numpy masks instead of bisect for descending pressure arrays
                mask_in_pbl = p_air >= p_pbl
                p_pbl_points = p_air[mask_in_pbl]
                ch4_pbl_points = ch4_air[mask_in_pbl]
                
                # Combine Tower + Aircraft points inside PBL for the spline
                p_for_spline = np.append(p0, p_pbl_points)
                ch4_for_spline = np.append(ch4_tower, ch4_pbl_points)
                
                # The "edge" is the highest aircraft measurement (lowest pressure) inside the PBL
                p_edge = p_pbl_points[-1] if p_pbl_points.size > 0 else p0
                ch4_edge = ch4_pbl_points[-1] if ch4_pbl_points.size > 0 else ch4_tower
                
                # Segment 2a: Spline from P0 to the aircraft edge
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, ch4_col_2a = fit_interp_safe(p_for_spline, ch4_for_spline, n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill from aircraft edge to PBL top
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, ch4_col_2b = fit_constant_gas(p_edge, p_pbl, ch4_edge, n_2b)
                
                p_col_2 = np.append(p_col_2a[::-1], p_col_2b)
                ch4_col_2 = np.append(ch4_col_2a[::-1], ch4_col_2b)
                
                # Sort upward (Surface to PBL Top)
                s_idx = np.argsort(p_col_2)
                p_col_2, ch4_col_2 = p_col_2[s_idx][::-1], ch4_col_2[s_idx][::-1]
            else:
                # If no aircraft in PBL, just fit P0 to PBL top
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                fill_val = ch4_air[0] if np.isnan(ch4_tower) else ch4_tower
                p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, fill_val, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
            # 1a: TP to CAMSinv top
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, ch4_col_1a = fit_interp_safe(p_cams[idx_01a:], ch4_cams[idx_01a:], n_1a, [p_tp, min(p_cams)])

            # 1b: Aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_air, ch4_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c2)
                p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                ch4_above = ch4_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_above, ch4_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, p_start_c, ch4_above[0], n_1c2)
                    p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                else:
                    p_col_1c, ch4_col_1c = np.array([]), np.array([])

            p_col_1 = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            ch4_col_1 = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
            s_idx_1 = np.argsort(p_col_1)
            p_col_1, ch4_col_1 = p_col_1[s_idx_1], ch4_col_1[s_idx_1]

            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_0': ch4_col_0,
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_0': p_col_0,
                'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                'Pressure_camsinv': p_cams, 'CH4_camsinv': ch4_cams,
                'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': ch4_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    return pd.DataFrame(all_results), failed

## Functions to generate GOSAT - Profile results (CH4)
def GOSAT_comparison_ch4(column_ch4, gosat):
    all_results = []
    failed = []
    
    # 1. Pre-process Aircraft Profiles
    column_ch4 = column_ch4.copy()
    profile_dict = {row.Date: row for row in column_ch4.itertuples(index=False)}
    
    available_dates = list(profile_dict.keys())
    if not available_dates:
        return pd.DataFrame(), ["No valid profile dates available."]

    for g_row in gosat.itertuples(index=False):
        try:
            g_date = g_row.Date
            
            # 2. Find Closest Profile Date
            closest_date = min(available_dates, key=lambda d: abs(d - g_date))
            if abs(closest_date - g_date) > timedelta(days=1):
                continue
            
            # Retrieve the complete profile namedtuple directly
            target = profile_dict[closest_date]

            # 3. Combine pressure and CH4 columns
            target_p = np.concatenate([target.Pressure_2[::-1], target.Pressure_1, target.Pressure_0])
            target_ch4 = np.concatenate([target.CH4_2[::-1], target.CH4_1, target.CH4_0])
            
            # 4. Binning
            bin_edges = np.linspace(0.1, g_row.P0, num=16)
            binned_means, _, _ = binned_statistic(target_p, target_ch4, statistic='mean', bins=bin_edges)
            binned_stds, _, _ = binned_statistic(target_p, target_ch4, statistic='std', bins=bin_edges)
            
            # 5. Vectorized NaN filling 
            s_means = pd.Series(binned_means)
            left = s_means.shift(1)
            right = s_means.shift(-1)
            neighbor_mean = pd.concat([left, right], axis=1).mean(axis=1)
            binned_means_filled = s_means.fillna(neighbor_mean).values
            binned_stds_filled = np.nan_to_num(binned_stds) 

            # 6. Vectorized Smoothing Calculation
            h_j = np.array(g_row.h_j)
            a_ch4_j = np.array(g_row.a_ch4_j)
            x_a = np.array(g_row.x_a)
            
            smth = h_j * a_ch4_j * (binned_means_filled - x_a * 1000)
            se = h_j * (1 - a_ch4_j) * (binned_means_filled - x_a * 1000)
            se_total = np.sum(h_j * (1 - a_ch4_j) * (binned_means_filled - x_a * 1000))
            deficit = np.sum(h_j * (binned_means_filled - x_a * 1000))
            smth_std = h_j * binned_stds_filled
            
            method2 = np.sum(smth) + g_row.xch4_a * 1000
            method2_std = np.sum(smth_std)
            
            # 7. Assembly
            all_results.append({
                'Date': closest_date, 'GOSAT_date': g_date, 'Year': target.Year,
                'Month': target.Date.month, 'Season': target.Season,
                'Pressure_all': target_p.tolist(), 'CH4_column': target_ch4.tolist(),
                'CH4_binned': binned_means_filled.tolist(), 'CH4_pripro': g_row.x_a,
                'Pressure_bins': bin_edges.tolist(), 'Pressure_binned': get_middle_pressure(bin_edges),
                'CH4_cak': method2, 'CH4_gosat': g_row.xch4_gosat, 'Diff': g_row.xch4_gosat - method2,
                'CH4_cak_std': method2_std, 'CAK': g_row.a_ch4_j, 'CH4_weighed': smth.tolist(), 
                'layer_se': se, 'total_se': se_total, 'deficit': deficit,
                'lat': g_row.lat, 'lon': g_row.lon, 'dist': g_row.dist,
                'CH4_tower': target.CH4_2[-1], 'CH4_tower_std': getattr(target, 'CH4_tower_std', np.nan),
                'temp_d': g_row.Temp_d, 'p0_d': g_row.P0_d, 'chi2': g_row.Chi2, 'xh2o': g_row.XH2O, 
                'dofs': g_row.DOFS, 'sanir': g_row.SA_NIR, 'saswir': g_row.SA_SWIR, 
                'sza': g_row.SZA, 'vza': g_row.VZA, 'am': g_row.AM, 'aot': g_row.AOT, 
                'snro2': g_row.SNR_O2, 'snrch4': g_row.SNR_CH4
            })
            
        except Exception as e:
            failed.append([g_row.Date, str(e)])
            
    if not all_results:
        return pd.DataFrame(), failed
        
    compare_df = pd.DataFrame(all_results)
    return compare_df, failed

## Tropopause perturbation analysis (CO2)
def CO2_column_tp_adj(column, camsinv, tp_rmse, direction):
    all_results = []
    failed = []
    
    # 1. Pre-align datasets to eliminate search overhead in the loop
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 2. Extract and sanitize variables
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            co2_air = np.array(row.CO2_column[1:])
            co2_tower = row.CO2_column[0]
            co2_cams = np.array(row.CO2_cams)
            
            # TP perturbation part
            this_tp_rmse = tp_rmse[row.Season-1]
            if direction == 1:
                p_tp += this_tp_rmse
            else:
                p_tp -= this_tp_rmse
            if len(row.CO2_column) < 5:
                failed.append([d, 'not enough obs'])
                continue

            # 3. Grid Setup
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                mask_in_pbl = p_air >= p_pbl
                p_pbl_pts = p_air[mask_in_pbl]
                co2_pbl_pts = co2_air[mask_in_pbl]
                
                p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                co2_edge = co2_pbl_pts[-1] if co2_pbl_pts.size > 0 else co2_tower
                
                # Segment 2a: Spline
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_col_2a, co2_col_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                                       np.append(co2_tower, co2_pbl_pts), 
                                                       n_2a, [p0, p_edge])
                
                # Segment 2b: Constant fill
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_col_2b, co2_col_2b = fit_constant_gas(p_edge, p_pbl, co2_edge, n_2b)
                
                # GLOBAL SORT UPWARD (Descending Pressure)
                p_col_2_unsorted = np.concatenate([p_col_2a, p_col_2b])
                co2_col_2_unsorted = np.concatenate([co2_col_2a, co2_col_2b])
                
                s_idx_2 = np.argsort(p_col_2_unsorted)[::-1] # High to Low
                p_col_2 = p_col_2_unsorted[s_idx_2]
                co2_col_2 = co2_col_2_unsorted[s_idx_2]
            else:
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
            # 1a: TP to model top (CAMSinv)
            idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
            n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
            p_col_1a, co2_col_1a = fit_interp_safe(p_cams[idx_01a:], co2_cams[idx_01a:], n_1a, [p_tp, model_top])

            # 1b: Highest aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)

            # 1c: PBL to highest aircraft
            if p_pbl > np.nanmax(p_air):
                # Interpolate across the aircraft range
                n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_col_1c1, co2_col_1c1 = fit_interp_safe(p_air, co2_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                # Fill PBL to lowest aircraft
                n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c2)
                p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
            else:
                mask_above_pbl = p_air < p_pbl
                p_above = p_air[mask_above_pbl]
                co2_above = co2_air[mask_above_pbl]
                
                if p_above.size > 0:
                    p_start_c = p_above[0]
                    n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, co2_col_1c1 = fit_interp_safe(p_above, co2_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                    p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, p_start_c, co2_above[0], n_1c2)
                    p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                else:
                    p_col_1c, co2_col_1c = np.array([]), np.array([])

            # GLOBAL SORT UPWARD (Descending Pressure)
            p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
            
            s_idx_1 = np.argsort(p_col_1_unsorted)[::-1] # High to Low
            p_col_1 = p_col_1_unsorted[s_idx_1]
            co2_col_1 = co2_col_1_unsorted[s_idx_1]
            
            all_results.append({
                'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1,
                'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                'Pressure_camsinv': p_cams, 'CO2_camsinv': co2_cams,
                'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
            })
        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results), failed

## Functions to generate GOSAT - Profile results (CO2)
def GOSAT_comparison_co2(column_co2, gosat):
    all_results = []
    failed = []
    
    # 1. Pre-process Aircraft Profiles
    column_co2 = column_co2.copy()
    profile_dict = {row.Date: row for row in column_co2.itertuples(index=False)}
    
    available_dates = list(profile_dict.keys())
    if not available_dates:
        return pd.DataFrame(), ["No valid profile dates available."]

    for g_row in gosat.itertuples(index=False):
        try:
            g_date = g_row.Date
            
            # 2. Find Closest Profile Date
            closest_date = min(available_dates, key=lambda d: abs(d - g_date))
            if abs(closest_date - g_date) > timedelta(days=1):
                continue
            
            # Retrieve the complete profile namedtuple directly
            target = profile_dict[closest_date]
            
            # 1. Combine pressure and CO2 columns using concatenation
            # target_p2[::-1] implies these are arrays/lists
            target_p = np.concatenate([target.Pressure_2[::-1], target.Pressure_1])
            target_co2 = np.concatenate([target.CO2_2[::-1], target.CO2_1])
            
            # 4. Binning
            bin_edges = np.linspace(0.1, g_row.P0, num=16)
            binned_means, _, _ = binned_statistic(target_p, target_co2, statistic='mean', bins=bin_edges)
            binned_stds, _, _ = binned_statistic(target_p, target_co2, statistic='std', bins=bin_edges)
                
            # 3. Vectorized NaN filling (mean of immediate neighbors)
            s_means = pd.Series(binned_means)
            left = s_means.shift(1)
            right = s_means.shift(-1)
            neighbor_mean = pd.concat([left, right], axis=1).mean(axis=1)
            binned_means_filled = s_means.fillna(neighbor_mean).values
                
            # Standard deviations usually aren't filled with neighbor means 
            # following original logic, so we just treat NaNs as 0 for sum
            binned_stds_filled = np.nan_to_num(binned_stds)

            # 4. Vectorized Smoothing Calculation
            h_j = np.array(g_row.h_j)
            a_co2_j = np.array(g_row.a_co2_j)
            x_a = np.array(g_row.x_a)
                
            # (binned_means_filled[i] - x_a[i])
            smth = h_j * a_co2_j * (binned_means_filled - x_a)
            smth_std = h_j * binned_stds_filled
            se = h_j * (1 - a_co2_j) * (binned_means_filled - x_a)
            se_total = np.sum(h_j * (1 - a_co2_j) * (binned_means_filled - x_a))            
            deficit = np.sum(h_j * (binned_means_filled - x_a))
                
            method2 = np.sum(smth) + g_row.xco2_a
            method2_std = np.sum(smth_std)
                
            # 5. Assemble Dictionary (matching exact original format)
            all_results.append({
                'Date': closest_date, 'GOSAT_date': g_date, 'Year': target.Year,
                'Month': target.Date.month, 'Season': target.Season,
                'Pressure_all': target_p.tolist(), 'CO2_column': target_co2.tolist(),
                'CO2_binned': binned_means_filled.tolist(), 'CO2_pripro': g_row.x_a,
                'Pressure_bins': bin_edges.tolist(), 'Pressure_binned': get_middle_pressure(bin_edges),
                'CO2_cak': method2, 'CO2_gosat': g_row.xco2_gosat, 'Diff': g_row.xco2_gosat - method2,
                'CO2_cak_std': method2_std, 'CAK': g_row.a_co2_j, 'CO2_weighed': smth.tolist(), 
                'CO2_tower_std': target.CO2_tower_std, 'layer_se': se, 'total_se': se_total, 'deficit': deficit,
                'p0_d': g_row.P0_d, 'chi2': g_row.Chi2,
                'sanir': g_row.SA_NIR, 'saswir': g_row.SA_SWIR, 'sza': g_row.SZA, 'vza': g_row.VZA, 'am': g_row.AM,
                'aot': g_row.AOT, 'dofs': g_row.DOFS, 'snrco2s': g_row.SNR_CO2s, 'snrco2w': g_row.SNR_CO2w,
                'lat': g_row.lat, 'lon': g_row.lon, 'dist': g_row.dist
            })
        except Exception as e:
            failed.append([g_row.Date, str(e)])
            
    if not all_results:
        return pd.DataFrame(), failed
        
    compare_df = pd.DataFrame(all_results)
    return compare_df, failed

## Functions to generate GOSAT - Profile results but CAK set to 1 (CO2)
def GOSAT_comparison_cakadj_co2(column_co2, gosat):
    all_results = []
    failed = []
    
    # 1. Pre-process Aircraft Profiles
    column_co2 = column_co2.copy()
    profile_dict = {row.Date: row for row in column_co2.itertuples(index=False)}
    
    available_dates = list(profile_dict.keys())
    if not available_dates:
        return pd.DataFrame(), ["No valid profile dates available."]

    for g_row in gosat.itertuples(index=False):
        try:
            g_date = g_row.Date
            
            # 2. Find Closest Profile Date
            closest_date = min(available_dates, key=lambda d: abs(d - g_date))
            if abs(closest_date - g_date) > timedelta(days=1):
                continue
            
            # Retrieve the complete profile namedtuple directly
            target = profile_dict[closest_date]
            
            # 1. Combine pressure and CO2 columns using concatenation
            # target_p2[::-1] implies these are arrays/lists
            target_p = np.concatenate([target.Pressure_2[::-1], target.Pressure_1])
            target_co2 = np.concatenate([target.CO2_2[::-1], target.CO2_1])
            
            # 4. Binning
            bin_edges = np.linspace(0.1, g_row.P0, num=16)
            binned_means, _, _ = binned_statistic(target_p, target_co2, statistic='mean', bins=bin_edges)
            binned_stds, _, _ = binned_statistic(target_p, target_co2, statistic='std', bins=bin_edges)
                
            # 3. Vectorized NaN filling (mean of immediate neighbors)
            s_means = pd.Series(binned_means)
            left = s_means.shift(1)
            right = s_means.shift(-1)
            neighbor_mean = pd.concat([left, right], axis=1).mean(axis=1)
            binned_means_filled = s_means.fillna(neighbor_mean).values
                
            # Standard deviations usually aren't filled with neighbor means 
            # following original logic, so we just treat NaNs as 0 for sum
            binned_stds_filled = np.nan_to_num(binned_stds)

            # 4. Vectorized Smoothing Calculation
            h_j = np.array(g_row.h_j)
            a_co2_j = np.array(g_row.a_co2_j)
            x_a = np.array(g_row.x_a)
            a_co2_j[a_co2_j <.7] = 1
                
            # (binned_means_filled[i] - x_a[i])
            smth = h_j * a_co2_j * (binned_means_filled - x_a)
            smth_std = h_j * binned_stds_filled
            se = h_j * (1 - a_co2_j) * (binned_means_filled - x_a)
            se_total = np.sum(h_j * (1 - a_co2_j) * (binned_means_filled - x_a))
                
            method2 = np.sum(smth) + g_row.xco2_a
            method2_std = np.sum(smth_std)
                
            # 5. Assemble Dictionary (matching exact original format)
            all_results.append({
                'Date': closest_date, 'GOSAT_date': g_date, 'Year': target.Year,
                'Month': target.Date.month, 'Season': target.Season,
                'Pressure_all': target_p.tolist(), 'CO2_column': target_co2.tolist(),
                'CO2_binned': binned_means_filled.tolist(), 'CO2_pripro': g_row.x_a,
                'Pressure_bins': bin_edges.tolist(), 'Pressure_binned': get_middle_pressure(bin_edges),
                'CO2_cak': method2, 'CO2_gosat': g_row.xco2_gosat, 'Diff': g_row.xco2_gosat - method2,
                'CO2_cak_std': method2_std, 'CAK': g_row.a_co2_j, 'CO2_weighed': smth.tolist(), 
                'CO2_tower_std': target.CO2_tower_std, 'layer_se': se, 'total_se': se_total,
                'p0_d': g_row.P0_d, 'chi2': g_row.Chi2,
                'sanir': g_row.SA_NIR, 'saswir': g_row.SA_SWIR, 'sza': g_row.SZA, 'vza': g_row.VZA, 'am': g_row.AM,
                'aot': g_row.AOT, 'dofs': g_row.DOFS, 'snrco2s': g_row.SNR_CO2s, 'snrco2w': g_row.SNR_CO2w,
                'lat': g_row.lat, 'lon': g_row.lon, 'dist': g_row.dist
            })
        except Exception as e:
            failed.append([g_row.Date, str(e)])
            
    if not all_results:
        return pd.DataFrame(), failed
        
    compare_df = pd.DataFrame(all_results)
    return compare_df, failed
## Check TP perturbation results
def compare_gosat_error_rms(original, adj, gas):
    if gas == 1:
        o_indexed = original.set_index('lat')
        a_indexed = adj.set_index('lat')
        diff = (o_indexed['CH4_cak'] - a_indexed['CH4_cak']).values
        diff_clean = diff[~np.isnan(diff)]
        diff_clean = diff_clean[np.isfinite(diff_clean)]
        sigma = np.sqrt(sum(diff_clean **2) / len(diff_clean))
    else:
        o_indexed = original.set_index('lat')
        a_indexed = adj.set_index('lat')
        diff = (o_indexed['CO2_cak'] - a_indexed['CO2_cak']).values
        diff_clean = diff[~np.isnan(diff)]
        diff_clean = diff_clean[np.isfinite(diff_clean)]
        sigma = np.sqrt(sum(diff_clean **2) / len(diff_clean))
    
    return sigma, len(diff_clean)

def long_error_rms(original, adj):
    o_indexed = original.set_index('Date')
    a_indexed = adj.set_index('Date')
    diff = (o_indexed['GHG_merra2'] - a_indexed['GHG_merra2']).values
    diff_clean = diff[~np.isnan(diff)]
    diff_clean = diff_clean[np.isfinite(diff_clean)]
    
    sigma = np.sqrt(sum(diff_clean **2) / len(diff_clean))
    
    return sigma, len(diff_clean)

## Tower CH4 adjustion test
def CH4_column_tower_adj(column, camsinv, adj_direction):
    all_results = []
    failed = []
    
    # Pre-align datasets
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 1. Variables and Scaling
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            ch4_air = np.array(row.CH4_column[1:]) - 5
            ch4_tower = row.CH4_column[0] - 5
            ch4_tower_std = row.CH4_tower_std
            ch4_cams = np.array(row.CH4_cams)

            # Tower adjustment part
            if math.isnan(ch4_tower_std):
                continue
            else:
                ch4_tower += ch4_tower_std * adj_direction
                
                if len(row.CH4_column) < 5:
                    failed.append([d, 'not enough obs'])
                    continue
    
                # 2. Grid Setup
                model_top = 0.1
                p_range_total = max(0.001, p0 - model_top)
                n_all = len(np.arange(model_top, p0, 0.01))
    
                ### TOP LEVEL: 0.5 to 0.1 hPa ###
                n_na = max(0, math.ceil(n_all * (min(p_cams) - model_top) / p_range_total))
                p_col_0, ch4_col_0 = fit_constant_gas(min(p_cams), model_top, ch4_cams[-1], n_na)
    
                ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
                # Logic: We use tower data and any aircraft data that falls below/within the PBL
                if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                    # Use numpy masks instead of bisect for descending pressure arrays
                    mask_in_pbl = p_air >= p_pbl
                    p_pbl_points = p_air[mask_in_pbl]
                    ch4_pbl_points = ch4_air[mask_in_pbl]
                    
                    # Combine Tower + Aircraft points inside PBL for the spline
                    p_for_spline = np.append(p0, p_pbl_points)
                    ch4_for_spline = np.append(ch4_tower, ch4_pbl_points)
                    
                    # The "edge" is the highest aircraft measurement (lowest pressure) inside the PBL
                    p_edge = p_pbl_points[-1] if p_pbl_points.size > 0 else p0
                    ch4_edge = ch4_pbl_points[-1] if ch4_pbl_points.size > 0 else ch4_tower
                    
                    # Segment 2a: Spline from P0 to the aircraft edge
                    n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                    p_col_2a, ch4_col_2a = fit_interp_safe(p_for_spline, ch4_for_spline, n_2a, [p0, p_edge])
                    
                    # Segment 2b: Constant fill from aircraft edge to PBL top
                    n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                    p_col_2b, ch4_col_2b = fit_constant_gas(p_edge, p_pbl, ch4_edge, n_2b)
                    
                    p_col_2 = np.append(p_col_2a[::-1], p_col_2b)
                    ch4_col_2 = np.append(ch4_col_2a[::-1], ch4_col_2b)
                    
                    # Sort upward (Surface to PBL Top)
                    s_idx = np.argsort(p_col_2)
                    p_col_2, ch4_col_2 = p_col_2[s_idx][::-1], ch4_col_2[s_idx][::-1]
                else:
                    # If no aircraft in PBL, just fit P0 to PBL top
                    n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                    fill_val = ch4_air[0] if np.isnan(ch4_tower) else ch4_tower
                    p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, fill_val, n_2)
    
                ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
                # 1a: TP to CAMSinv top
                idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
                n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
                p_col_1a, ch4_col_1a = fit_interp_safe(p_cams[idx_01a:], ch4_cams[idx_01a:], n_1a, [p_tp, min(p_cams)])
    
                # 1b: Aircraft to TP
                n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
                p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)
    
                # 1c: PBL to highest aircraft
                if p_pbl > np.nanmax(p_air):
                    n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_air, ch4_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                    p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c2)
                    p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                else:
                    mask_above_pbl = p_air < p_pbl
                    p_above = p_air[mask_above_pbl]
                    ch4_above = ch4_air[mask_above_pbl]
                    
                    if p_above.size > 0:
                        p_start_c = p_above[0]
                        n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                        p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_above, ch4_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                        n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                        p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, p_start_c, ch4_above[0], n_1c2)
                        p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                    else:
                        p_col_1c, ch4_col_1c = np.array([]), np.array([])
    
                p_col_1 = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
                ch4_col_1 = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
                s_idx_1 = np.argsort(p_col_1)
                p_col_1, ch4_col_1 = p_col_1[s_idx_1], ch4_col_1[s_idx_1]
    
                all_results.append({
                    'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                    'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                    'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_0': ch4_col_0,
                    'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_0': p_col_0,
                    'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                    'Pressure_camsinv': p_cams, 'CH4_camsinv': ch4_cams,
                    'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': ch4_tower_std
                })
        except Exception as e:
            failed.append([d, str(e)])

    return pd.DataFrame(all_results), failed

## Tower CO2 adjustion test
def CO2_column_tower_adj(column, camsinv, adj_direction):
    all_results = []
    failed = []
    
    # 1. Pre-align datasets to eliminate search overhead in the loop
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 2. Extract variables
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            co2_air = np.array(row.CO2_column[1:])
            co2_tower = row.CO2_column[0]
            co2_tower_std = row.CO2_tower_std
            co2_cams = np.array(row.CO2_cams)

            if math.isnan(co2_tower_std):
                continue
            else:
                co2_tower += co2_tower_std * adj_direction
                
                if len(row.CO2_column) < 5:
                    failed.append([d, 'not enough obs'])
                    continue
    
                # 3. Grid Setup
                model_top = 0.1
                p_range_total = max(0.001, p0 - model_top)
                n_all = len(np.arange(model_top, p0, 0.01))
    
                ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
                if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                    mask_in_pbl = p_air >= p_pbl
                    p_pbl_pts = p_air[mask_in_pbl]
                    co2_pbl_pts = co2_air[mask_in_pbl]
                    
                    p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                    co2_edge = co2_pbl_pts[-1] if co2_pbl_pts.size > 0 else co2_tower
                    
                    # Segment 2a: Spline
                    n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                    p_col_2a, co2_col_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                                           np.append(co2_tower, co2_pbl_pts), 
                                                           n_2a, [p0, p_edge])
                    
                    # Segment 2b: Constant fill
                    n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                    p_col_2b, co2_col_2b = fit_constant_gas(p_edge, p_pbl, co2_edge, n_2b)
                    
                    # GLOBAL SORT UPWARD (Descending Pressure)
                    p_col_2_unsorted = np.concatenate([p_col_2a, p_col_2b])
                    co2_col_2_unsorted = np.concatenate([co2_col_2a, co2_col_2b])
                    
                    s_idx_2 = np.argsort(p_col_2_unsorted)[::-1] # High to Low
                    p_col_2 = p_col_2_unsorted[s_idx_2]
                    co2_col_2 = co2_col_2_unsorted[s_idx_2]
                else:
                    n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                    p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)
    
                ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
                # 1a: TP to model top (CAMSinv)
                idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
                n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
                p_col_1a, co2_col_1a = fit_interp_safe(p_cams[idx_01a:], co2_cams[idx_01a:], n_1a, [p_tp, model_top])
    
                # 1b: Highest aircraft to TP
                n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
                p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)
    
                # 1c: PBL to highest aircraft
                if p_pbl > np.nanmax(p_air):
                    # Interpolate across the aircraft range
                    n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, co2_col_1c1 = fit_interp_safe(p_air, co2_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                    # Fill PBL to lowest aircraft
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                    p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c2)
                    p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                else:
                    mask_above_pbl = p_air < p_pbl
                    p_above = p_air[mask_above_pbl]
                    co2_above = co2_air[mask_above_pbl]
                    
                    if p_above.size > 0:
                        p_start_c = p_above[0]
                        n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                        p_col_1c1, co2_col_1c1 = fit_interp_safe(p_above, co2_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                        n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                        p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, p_start_c, co2_above[0], n_1c2)
                        p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                    else:
                        p_col_1c, co2_col_1c = np.array([]), np.array([])
    
                # GLOBAL SORT UPWARD (Descending Pressure)
                p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
                co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
                
                s_idx_1 = np.argsort(p_col_1_unsorted)[::-1] # High to Low
                p_col_1 = p_col_1_unsorted[s_idx_1]
                co2_col_1 = co2_col_1_unsorted[s_idx_1]
                
                all_results.append({
                    'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                    'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                    'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 
                    'Pressure_2': p_col_2, 'Pressure_1': p_col_1,
                    'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                    'Pressure_camsinv': p_cams, 'CO2_camsinv': co2_cams,
                    'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
                })
        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results), failed

## Tower CH4 missing test
def CH4_column_tower_missing(column, camsinv):
    all_results = []
    failed = []
    
    # Pre-align datasets
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 1. Variables and Scaling
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            ch4_air = np.array(row.CH4_column[1:]) - 5
            ch4_tower = row.CH4_column[0] - 5
            ch4_tower_std = row.CH4_tower_std
            ch4_cams = np.array(row.CH4_cams)

            if np.isnan(ch4_tower_std):
                continue
            else:
                ch4_tower = ch4_air[0]
                
                if len(row.CH4_column) < 5:
                    failed.append([d, 'not enough obs'])
                    continue
    
                # 2. Grid Setup
                model_top = 0.1
                p_range_total = max(0.001, p0 - model_top)
                n_all = len(np.arange(model_top, p0, 0.01))
    
                ### TOP LEVEL: 0.5 to 0.1 hPa ###
                n_na = max(0, math.ceil(n_all * (min(p_cams) - model_top) / p_range_total))
                p_col_0, ch4_col_0 = fit_constant_gas(min(p_cams), model_top, ch4_cams[-1], n_na)
    
                ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
                # Logic: We use tower data and any aircraft data that falls below/within the PBL
                if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                    # Use numpy masks instead of bisect for descending pressure arrays
                    mask_in_pbl = p_air >= p_pbl
                    p_pbl_points = p_air[mask_in_pbl]
                    ch4_pbl_points = ch4_air[mask_in_pbl]
                    
                    # Combine Tower + Aircraft points inside PBL for the spline
                    p_for_spline = np.append(p0, p_pbl_points)
                    ch4_for_spline = np.append(ch4_tower, ch4_pbl_points)
                    
                    # The "edge" is the highest aircraft measurement (lowest pressure) inside the PBL
                    p_edge = p_pbl_points[-1] if p_pbl_points.size > 0 else p0
                    ch4_edge = ch4_pbl_points[-1] if ch4_pbl_points.size > 0 else ch4_tower
                    
                    # Segment 2a: Spline from P0 to the aircraft edge
                    n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                    p_col_2a, ch4_col_2a = fit_interp_safe(p_for_spline, ch4_for_spline, n_2a, [p0, p_edge])
                    
                    # Segment 2b: Constant fill from aircraft edge to PBL top
                    n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                    p_col_2b, ch4_col_2b = fit_constant_gas(p_edge, p_pbl, ch4_edge, n_2b)
                    
                    p_col_2 = np.append(p_col_2a[::-1], p_col_2b)
                    ch4_col_2 = np.append(ch4_col_2a[::-1], ch4_col_2b)
                    
                    # Sort upward (Surface to PBL Top)
                    s_idx = np.argsort(p_col_2)
                    p_col_2, ch4_col_2 = p_col_2[s_idx][::-1], ch4_col_2[s_idx][::-1]
                else:
                    # If no aircraft in PBL, just fit P0 to PBL top
                    n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                    fill_val = ch4_air[0] if np.isnan(ch4_tower) else ch4_tower
                    p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, fill_val, n_2)
    
                ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
                # 1a: TP to CAMSinv top
                idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
                n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
                p_col_1a, ch4_col_1a = fit_interp_safe(p_cams[idx_01a:], ch4_cams[idx_01a:], n_1a, [p_tp, min(p_cams)])
    
                # 1b: Aircraft to TP
                n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
                p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)
    
                # 1c: PBL to highest aircraft
                if p_pbl > np.nanmax(p_air):
                    n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_air, ch4_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                    p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c2)
                    p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                else:
                    mask_above_pbl = p_air < p_pbl
                    p_above = p_air[mask_above_pbl]
                    ch4_above = ch4_air[mask_above_pbl]
                    
                    if p_above.size > 0:
                        p_start_c = p_above[0]
                        n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                        p_col_1c1, ch4_col_1c1 = fit_interp_safe(p_above, ch4_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                        n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                        p_col_1c2, ch4_col_1c2 = fit_constant_gas(p_pbl, p_start_c, ch4_above[0], n_1c2)
                        p_col_1c, ch4_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(ch4_col_1c2, ch4_col_1c1)
                    else:
                        p_col_1c, ch4_col_1c = np.array([]), np.array([])
    
                p_col_1 = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
                ch4_col_1 = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
                s_idx_1 = np.argsort(p_col_1)
                p_col_1, ch4_col_1 = p_col_1[s_idx_1], ch4_col_1[s_idx_1]
    
                all_results.append({
                    'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                    'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                    'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_0': ch4_col_0,
                    'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_0': p_col_0,
                    'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                    'Pressure_camsinv': p_cams, 'CH4_camsinv': ch4_cams,
                    'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': ch4_tower_std
                })
        except Exception as e:
            failed.append([d, str(e)])

    return pd.DataFrame(all_results), failed

## Tower CO2 missing test
def CO2_column_tower_missing(column, camsinv):
    all_results = []
    failed = []
    
    # 1. Pre-align datasets to eliminate search overhead in the loop
    combined = pd.merge(
        column, 
        camsinv[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'}),
        on='Date', how='inner'
    )

    for _, row in combined.iterrows():
        d = row.Date
        try:
            # 2. Extract and sanitize variables
            p0 = row.P0
            p_pbl = row.PBL_Pressure
            p_tp = row.TP_B
            p_air = np.array(row.Pressure[1:])
            p_cams = np.array(row.P_cams)
            
            co2_air = np.array(row.CO2_column[1:])
            co2_tower = row.CO2_column[0]
            co2_tower_std = row.CO2_tower_std
            co2_cams = np.array(row.CO2_cams)

            if np.isnan(co2_tower_std):
                continue
            else:
                co2_tower = co2_air[0]
                
                if len(row.CO2_column) < 5:
                    failed.append([d, 'not enough obs'])
                    continue
    
                # 3. Grid Setup
                model_top = 0.1
                p_range_total = max(0.001, p0 - model_top)
                n_all = len(np.arange(model_top, p0, 0.01))
    
                ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
                if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                    mask_in_pbl = p_air >= p_pbl
                    p_pbl_pts = p_air[mask_in_pbl]
                    co2_pbl_pts = co2_air[mask_in_pbl]
                    
                    p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                    co2_edge = co2_pbl_pts[-1] if co2_pbl_pts.size > 0 else co2_tower
                    
                    # Segment 2a: Spline
                    n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                    p_col_2a, co2_col_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                                           np.append(co2_tower, co2_pbl_pts), 
                                                           n_2a, [p0, p_edge])
                    
                    # Segment 2b: Constant fill
                    n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                    p_col_2b, co2_col_2b = fit_constant_gas(p_edge, p_pbl, co2_edge, n_2b)
                    
                    # GLOBAL SORT UPWARD (Descending Pressure)
                    p_col_2_unsorted = np.concatenate([p_col_2a, p_col_2b])
                    co2_col_2_unsorted = np.concatenate([co2_col_2a, co2_col_2b])
                    
                    s_idx_2 = np.argsort(p_col_2_unsorted)[::-1] # High to Low
                    p_col_2 = p_col_2_unsorted[s_idx_2]
                    co2_col_2 = co2_col_2_unsorted[s_idx_2]
                else:
                    n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                    p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)
    
                ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
                # 1a: TP to model top (CAMSinv)
                idx_01a = max(0, (np.abs(p_cams - p_tp)).argmin() - 1)
                n_1a = max(0, math.ceil(n_all * (max(p_cams[idx_01a:]) - model_top) / p_range_total))
                p_col_1a, co2_col_1a = fit_interp_safe(p_cams[idx_01a:], co2_cams[idx_01a:], n_1a, [p_tp, model_top])
    
                # 1b: Highest aircraft to TP
                n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
                p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)
    
                # 1c: PBL to highest aircraft
                if p_pbl > np.nanmax(p_air):
                    # Interpolate across the aircraft range
                    n_1c1 = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                    p_col_1c1, co2_col_1c1 = fit_interp_safe(p_air, co2_air, n_1c1, [np.nanmax(p_air), np.nanmin(p_air)])
                    # Fill PBL to lowest aircraft
                    n_1c2 = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                    p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c2)
                    p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                else:
                    mask_above_pbl = p_air < p_pbl
                    p_above = p_air[mask_above_pbl]
                    co2_above = co2_air[mask_above_pbl]
                    
                    if p_above.size > 0:
                        p_start_c = p_above[0]
                        n_1c1 = max(0, math.ceil(n_all * (p_start_c - np.nanmin(p_air)) / p_range_total))
                        p_col_1c1, co2_col_1c1 = fit_interp_safe(p_above, co2_above, n_1c1, [p_start_c, np.nanmin(p_air)])
                        n_1c2 = max(0, math.ceil(n_all * (p_pbl - p_start_c) / p_range_total))
                        p_col_1c2, co2_col_1c2 = fit_constant_gas(p_pbl, p_start_c, co2_above[0], n_1c2)
                        p_col_1c, co2_col_1c = np.append(p_col_1c2, p_col_1c1), np.append(co2_col_1c2, co2_col_1c1)
                    else:
                        p_col_1c, co2_col_1c = np.array([]), np.array([])
    
                # GLOBAL SORT UPWARD (Descending Pressure)
                p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
                co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
                
                s_idx_1 = np.argsort(p_col_1_unsorted)[::-1] # High to Low
                p_col_1 = p_col_1_unsorted[s_idx_1]
                co2_col_1 = co2_col_1_unsorted[s_idx_1]
                
                all_results.append({
                    'Date': d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                    'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                    'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 
                    'Pressure_2': p_col_2, 'Pressure_1': p_col_1,
                    'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                    'Pressure_camsinv': p_cams, 'CO2_camsinv': co2_cams,
                    'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
                })
        except Exception as e:
            failed.append([d, str(e)])

    if not all_results:
        return pd.DataFrame(), failed

    return pd.DataFrame(all_results), failed

## Stratosphere adjustion test
def create_hybrid_profile(c_p, c_ghg, s_p, s_ghg, model_top=0.1):    
    cp, cv = np.asarray(c_p, dtype=float), np.asarray(c_ghg, dtype=float)
    sp, sv = np.asarray(s_p, dtype=float), np.asarray(s_ghg, dtype=float)

    # 1. Ensure CAMS is sorted descending by pressure (Surface -> Space)
    c_sort = np.argsort(cp)[::-1]
    cp, cv = cp[c_sort], cv[c_sort]
    
    # 2. STRICT SCISAT FILTERING
    valid_sci = np.isfinite(sp) & np.isfinite(sv) & (sv > 1.0) & (sp >= model_top)
    sp_clean = sp[valid_sci]
    sv_clean = sv[valid_sci]

    # 3. Early exit if SCISAT is insufficient AFTER cleaning
    if len(sp_clean) < 5:
        return cp.tolist(), cv.tolist()
    
    # 4. Safe boundary generation
    p_min, p_max = np.min(sp_clean), np.max(sp_clean)
    
    # Keep CAMS data outside the SCISAT bounds
    keep_mask = (cp < p_min) | (cp > p_max)
    
    # 5. Merge and sort safely
    new_p = np.concatenate([cp[keep_mask], sp_clean])
    new_v = np.concatenate([cv[keep_mask], sv_clean])
    
    sort_idx = np.argsort(new_p)[::-1]
    return new_p[sort_idx].tolist(), new_v[sort_idx].tolist()

def CH4_strat_adjust(column, scisat, camsinv):
    failed, all_results = [], []
    
    cams_sub = camsinv[['Date', 'Pressure', 'CH4']].rename(columns={'Pressure': 'P_cams', 'CH4': 'CH4_cams'})
    sci_sub = scisat[['Date', 'Pressure_CH4', 'CH4']].rename(columns={'Pressure_CH4': 'P_sci', 'CH4': 'CH4_sci'})
    
    combined = pd.merge(column, cams_sub, on='Date', how='inner')
    combined = pd.merge(combined, sci_sub, on='Date', how='inner')

    for row in combined.itertuples(index=False):
        d = row.Date
        try:
            p0, p_pbl, p_tp = row.P0, row.PBL_Pressure, row.TP_B
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            p_air_raw = np.array(row.Pressure[1:], dtype=float)
            ch4_air_raw = np.array(row.CH4_column[1:], dtype=float) - 5.0
            
            # Immediately strip out NaNs and non-physical values before [-1] indexing
            valid_air = np.isfinite(p_air_raw) & np.isfinite(ch4_air_raw)
            p_air = p_air_raw[valid_air]
            ch4_air = ch4_air_raw[valid_air]
            
            ch4_tower = row.CH4_column[0] - 5.0
            if (not np.isfinite(ch4_tower) or ch4_tower <= 100) and p_air.size > 0:
                ch4_tower = ch4_air[0]

            if p_air.size < 5:
                failed.append([d, 'insufficient valid CH4 obs']); continue

            ### BOTTOM LEVEL: Surface to PBL (CH4_2) ###
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                mask_in_pbl = p_air >= p_pbl
                p_pbl_pts, ch4_pbl_pts = p_air[mask_in_pbl], ch4_air[mask_in_pbl]
                
                p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_2a, ch4_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                               np.append(ch4_tower, ch4_pbl_pts), 
                                               n_2a, [p0, p_edge])
                
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_2b, ch4_2b = fit_constant_gas(p_edge, p_pbl, ch4_pbl_pts[-1] if p_pbl_pts.size > 0 else ch4_tower, n_2b)
                
                p_col_2_unsorted = np.concatenate([p_2a, p_2b])
                ch4_col_2_unsorted = np.concatenate([ch4_2a, ch4_2b])
                s_idx_2 = np.argsort(p_col_2_unsorted)[::-1]
                p_col_2, ch4_col_2 = p_col_2_unsorted[s_idx_2], ch4_col_2_unsorted[s_idx_2]
            else:
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                p_col_2, ch4_col_2 = fit_constant_gas(p0, p_pbl, ch4_tower, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CH4_1) ###
            p_hybrid, ch4_hybrid = create_hybrid_profile(row.P_cams, row.CH4_cams, row.P_sci, row.CH4_sci, model_top)
            
            p_hy_arr = np.array(p_hybrid, dtype=float)
            ch4_hybrid = np.array(ch4_hybrid, dtype=float)
            
            # 1a: TP to Model Top (Hybrid profile is already clean)
            idx_tp = np.abs(p_hy_arr - p_tp).argmin()
            n_1a = max(0, math.ceil(n_all * (p_tp - model_top) / p_range_total))
            p_col_1a, ch4_col_1a = fit_interp_safe(p_hy_arr[idx_tp:], ch4_hybrid[idx_tp:], n_1a, [p_tp, model_top])

            # 1b: Highest aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, ch4_col_1b = fit_constant_gas(p_air[-1], p_tp, ch4_air[-1], n_1b)
            
            # 1c: PBL to highest aircraft (Spline + Gap fill)
            if p_pbl > np.nanmax(p_air):
                n_1c_spline = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_1c1, ch4_1c1 = fit_interp_safe(p_air, ch4_air, n_1c_spline, [np.nanmax(p_air), np.nanmin(p_air)])
                n_1c_gap = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_1c2, ch4_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), ch4_air[0], n_1c_gap)
                p_col_1c, ch4_col_1c = np.append(p_1c2, p_1c1), np.append(ch4_1c2, ch4_1c1)
            else:
                sorted_p_air = np.sort(p_air)
                idx_p12 = len(p_air) - bisect.bisect(sorted_p_air, p_pbl)
                p_start_node = p_air[idx_p12]
                n_1c_spline = max(0, math.ceil(n_all * (p_start_node - np.nanmin(p_air)) / p_range_total))
                p_1c1, ch4_1c1 = fit_interp_safe(p_air[idx_p12:], ch4_air[idx_p12:], n_1c_spline, [p_start_node, np.nanmin(p_air)])
                n_1c_gap = max(0, math.ceil(n_all * (p_pbl - p_start_node) / p_range_total))
                p_1c2, ch4_1c2 = fit_constant_gas(p_pbl, p_start_node, ch4_air[idx_p12], n_1c_gap)
                p_col_1c, ch4_col_1c = np.append(p_1c2, p_1c1), np.append(ch4_1c2, ch4_1c1)
            
            # Combine and Sort
            p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            ch4_col_1_unsorted = np.concatenate([ch4_col_1c, ch4_col_1b, ch4_col_1a])
            
            # Final validation check before appending
            valid_final = np.isfinite(ch4_col_1_unsorted) & (ch4_col_1_unsorted > 100)
            p_col_1_unsorted = p_col_1_unsorted[valid_final]
            ch4_col_1_unsorted = ch4_col_1_unsorted[valid_final]
            
            s_idx_1 = np.argsort(p_col_1_unsorted)[::-1]
            p_col_1, ch4_col_1 = p_col_1_unsorted[s_idx_1], ch4_col_1_unsorted[s_idx_1]

            all_results.append({
                "Date": d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CH4_2': ch4_col_2, 'CH4_1': ch4_col_1, 'CH4_scisat': np.array(row.CH4_sci),
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_scisat': np.array(row.P_sci),
                'Pressure_aircraft': p_air, 'CH4_aircraft': ch4_air,
                'Pressure_camsinv': np.array(row.P_cams), 'CH4_camsinv': np.array(row.CH4_cams),
                'Pressure_tower': p0, 'CH4_tower': ch4_tower, 'CH4_tower_std': row.CH4_tower_std
            })

        except Exception as e:
            failed.append([d, str(e)])

    if not all_results: return pd.DataFrame(), failed
    return pd.DataFrame(all_results).sort_values('Date').reset_index(drop=True), failed

def CO2_strat_adjust(column, scisat, camsinv):
    failed, all_results = [], []
    
    cams_sub = camsinv[['Date', 'Pressure', 'CO2']].rename(columns={'Pressure': 'P_cams', 'CO2': 'CO2_cams'})
    sci_sub = scisat[['Date', 'Pressure_CO2', 'CO2']].rename(columns={'Pressure_CO2': 'P_sci', 'CO2': 'CO2_sci'})
    
    combined = pd.merge(column, cams_sub, on='Date', how='inner')
    combined = pd.merge(combined, sci_sub, on='Date', how='inner')

    for row in combined.itertuples(index=False):
        d = row.Date
        try:
            p0, p_pbl, p_tp = row.P0, row.PBL_Pressure, row.TP_B
            model_top = 0.1
            p_range_total = max(0.001, p0 - model_top)
            n_all = len(np.arange(model_top, p0, 0.01))

            # --- ROBUST DATA CLEANING ---
            p_air_raw = np.array(row.Pressure[1:], dtype=float)
            co2_air_raw = np.array(row.CO2_column[1:], dtype=float)
            
            # Immediately strip out NaNs and non-physical values before [-1] indexing
            valid_air = np.isfinite(p_air_raw) & np.isfinite(co2_air_raw) & (co2_air_raw > 100)
            p_air = p_air_raw[valid_air]
            co2_air = co2_air_raw[valid_air]
            
            co2_tower = row.CO2_column[0]
            if (not np.isfinite(co2_tower) or co2_tower <= 100) and p_air.size > 0:
                co2_tower = co2_air[0]

            if p_air.size < 5:
                failed.append([d, 'insufficient valid CH4 obs']); continue

            ### BOTTOM LEVEL: Surface to PBL (CO2_2) ###
            if p_air.size > 0 and np.nanmax(p_air) > p_pbl:
                mask_in_pbl = p_air >= p_pbl
                p_pbl_pts, co2_pbl_pts = p_air[mask_in_pbl], co2_air[mask_in_pbl]
                
                p_edge = p_pbl_pts[-1] if p_pbl_pts.size > 0 else p0
                n_2a = max(0, math.ceil(n_all * (p0 - p_edge) / p_range_total))
                p_2a, co2_2a = fit_interp_safe(np.append(p0, p_pbl_pts), 
                                               np.append(co2_tower, co2_pbl_pts), 
                                               n_2a, [p0, p_edge])
                
                n_2b = max(0, math.ceil(n_all * (p_edge - p_pbl) / p_range_total))
                p_2b, co2_2b = fit_constant_gas(p_edge, p_pbl, co2_pbl_pts[-1] if p_pbl_pts.size > 0 else co2_tower, n_2b)
                
                p_col_2_unsorted = np.concatenate([p_2a, p_2b])
                co2_col_2_unsorted = np.concatenate([co2_2a, co2_2b])
                s_idx_2 = np.argsort(p_col_2_unsorted)[::-1]
                p_col_2, co2_col_2 = p_col_2_unsorted[s_idx_2], co2_col_2_unsorted[s_idx_2]
            else:
                n_2 = max(0, math.ceil(n_all * (p0 - p_pbl) / p_range_total))
                p_col_2, co2_col_2 = fit_constant_gas(p0, p_pbl, co2_tower, n_2)

            ### MIDDLE LEVEL: PBL to Model Top (CO2_1) ###
            p_hybrid, co2_hybrid = create_hybrid_profile(row.P_cams, row.CO2_cams, row.P_sci, row.CO2_sci, model_top)
            
            p_hy_arr = np.array(p_hybrid, dtype=float)
            co2_hybrid = np.array(co2_hybrid, dtype=float)
            
            # 1a: TP to Model Top (Hybrid profile is already clean)
            idx_tp = np.abs(p_hy_arr - p_tp).argmin()
            n_1a = max(0, math.ceil(n_all * (p_tp - model_top) / p_range_total))
            p_col_1a, co2_col_1a = fit_interp_safe(p_hy_arr[idx_tp:], co2_hybrid[idx_tp:], n_1a, [p_tp, model_top])

            # 1b: Highest aircraft to TP
            n_1b = max(0, math.ceil(n_all * (p_air[-1] - p_tp) / p_range_total))
            p_col_1b, co2_col_1b = fit_constant_gas(p_air[-1], p_tp, co2_air[-1], n_1b)
            
            # 1c: PBL to highest aircraft (Spline + Gap fill)
            if p_pbl > np.nanmax(p_air):
                n_1c_spline = max(0, math.ceil(n_all * (np.nanmax(p_air) - np.nanmin(p_air)) / p_range_total))
                p_1c1, co2_1c1 = fit_interp_safe(p_air, co2_air, n_1c_spline, [np.nanmax(p_air), np.nanmin(p_air)])
                n_1c_gap = max(0, math.ceil(n_all * (p_pbl - np.nanmax(p_air)) / p_range_total))
                p_1c2, co2_1c2 = fit_constant_gas(p_pbl, np.nanmax(p_air), co2_air[0], n_1c_gap)
                p_col_1c, co2_col_1c = np.append(p_1c2, p_1c1), np.append(co2_1c2, co2_1c1)
            else:
                sorted_p_air = np.sort(p_air)
                idx_p12 = len(p_air) - bisect.bisect(sorted_p_air, p_pbl)
                p_start_node = p_air[idx_p12]
                n_1c_spline = max(0, math.ceil(n_all * (p_start_node - np.nanmin(p_air)) / p_range_total))
                p_1c1, co2_1c1 = fit_interp_safe(p_air[idx_p12:], co2_air[idx_p12:], n_1c_spline, [p_start_node, np.nanmin(p_air)])
                n_1c_gap = max(0, math.ceil(n_all * (p_pbl - p_start_node) / p_range_total))
                p_1c2, co2_1c2 = fit_constant_gas(p_pbl, p_start_node, co2_air[idx_p12], n_1c_gap)
                p_col_1c, co2_col_1c = np.append(p_1c2, p_1c1), np.append(co2_1c2, co2_1c1)
            
            # Combine and Sort
            p_col_1_unsorted = np.concatenate([p_col_1c, p_col_1b, p_col_1a])
            co2_col_1_unsorted = np.concatenate([co2_col_1c, co2_col_1b, co2_col_1a])
            
            # Final validation check before appending
            valid_final = np.isfinite(co2_col_1_unsorted) & (co2_col_1_unsorted > 100)
            p_col_1_unsorted = p_col_1_unsorted[valid_final]
            co2_col_1_unsorted = co2_col_1_unsorted[valid_final]
            
            s_idx_1 = np.argsort(p_col_1_unsorted)[::-1]
            p_col_1, co2_col_1 = p_col_1_unsorted[s_idx_1], co2_col_1_unsorted[s_idx_1]

            all_results.append({
                "Date": d, 'Year': row.Year, 'Season': row.Season, 'Hour': row.Hour,
                'Pressure_PBL': p_pbl, 'Pressure_TP': p_tp,
                'CO2_2': co2_col_2, 'CO2_1': co2_col_1, 'CO2_scisat': np.array(row.CO2_sci),
                'Pressure_2': p_col_2, 'Pressure_1': p_col_1, 'Pressure_scisat': np.array(row.P_sci),
                'Pressure_aircraft': p_air, 'CO2_aircraft': co2_air,
                'Pressure_camsinv': np.array(row.P_cams), 'CO2_camsinv': np.array(row.CO2_cams),
                'Pressure_tower': p0, 'CO2_tower': co2_tower, 'CO2_tower_std': row.CO2_tower_std
            })

        except Exception as e:
            failed.append([d, str(e)])

    if not all_results: return pd.DataFrame(), failed
    return pd.DataFrame(all_results).sort_values('Date').reset_index(drop=True), failed

def GOSAT_comparison_str_adj_ch4(column_ch4, gosat):
    all_results = []
    failed = []
    
    # 1. Pre-process Aircraft Profiles
    column_ch4 = column_ch4.copy()
    profile_dict = {row.Date: row for row in column_ch4.itertuples(index=False)}
    
    available_dates = list(profile_dict.keys())
    if not available_dates:
        return pd.DataFrame(), ["No valid profile dates available."]

    for g_row in gosat.itertuples(index=False):
        try:
            g_date = g_row.Date
            
            # 2. Find Closest Profile Date
            closest_date = min(available_dates, key=lambda d: abs(d - g_date))
            if abs(closest_date - g_date) > timedelta(days=1):
                continue
            
            # Retrieve the complete profile namedtuple directly
            target = profile_dict[closest_date]

            # 3. Combine pressure and CH4 columns
            target_p = np.concatenate([target.Pressure_2[::-1], target.Pressure_1])
            target_ch4 = np.concatenate([target.CH4_2[::-1], target.CH4_1])
            
            # 4. Binning
            bin_edges = np.linspace(0.1, g_row.P0, num=16)
            binned_means, _, _ = binned_statistic(target_p, target_ch4, statistic='mean', bins=bin_edges)
            binned_stds, _, _ = binned_statistic(target_p, target_ch4, statistic='std', bins=bin_edges)
            
            # 5. Vectorized NaN filling 
            s_means = pd.Series(binned_means)
            left = s_means.shift(1)
            right = s_means.shift(-1)
            neighbor_mean = pd.concat([left, right], axis=1).mean(axis=1)
            binned_means_filled = s_means.fillna(neighbor_mean).values
            binned_stds_filled = np.nan_to_num(binned_stds) 

            # 6. Vectorized Smoothing Calculation (CAK)
            h_j = np.array(g_row.h_j)
            a_ch4_j = np.array(g_row.a_ch4_j)
            x_a = np.array(g_row.x_a)
            
            smth = h_j * a_ch4_j * (binned_means_filled - x_a * 1000)
            se = h_j * (1 - a_ch4_j) * (binned_means_filled - x_a * 1000)
            smth_std = h_j * binned_stds_filled
            
            method2 = np.sum(smth) + g_row.xch4_a * 1000
            method2_std = np.sum(smth_std)
            
            # 7. Assembly
            all_results.append({
                'Date': closest_date, 'GOSAT_date': g_date, 'Year': target.Year,
                'Month': target.Date.month, 'Season': target.Season,
                'Pressure_all': target_p.tolist(), 'CH4_column': target_ch4.tolist(),
                'CH4_binned': binned_means_filled.tolist(), 'CH4_pripro': g_row.x_a,
                'Pressure_bins': bin_edges.tolist(), 'Pressure_binned': get_middle_pressure(bin_edges),
                'CH4_cak': method2, 'CH4_gosat': g_row.xch4_gosat, 'Diff': g_row.xch4_gosat - method2,
                'CH4_cak_std': method2_std, 'CAK': g_row.a_ch4_j, 'CH4_weighed': smth.tolist(), 
                'layer_se': se, 'total_se': float(np.sum(se)),
                'lat': g_row.lat, 'lon': g_row.lon, 'dist': g_row.dist,
                'CH4_tower': target.CH4_2[-1], 'CH4_tower_std': getattr(target, 'CH4_tower_std', np.nan),
                'temp_d': g_row.Temp_d, 'p0_d': g_row.P0_d, 'chi2': g_row.Chi2, 'xh2o': g_row.XH2O, 
                'dofs': g_row.DOFS, 'sza': g_row.SZA, 'vza': g_row.VZA, 'am': g_row.AM, 'aot': g_row.AOT, 
                'snro2': g_row.SNR_O2, 'snrch4': g_row.SNR_CH4
            })
            
        except Exception as e:
            failed.append([g_row.Date, str(e)])
            
    if not all_results:
        return pd.DataFrame(), failed
        
    compare_df = pd.DataFrame(all_results)
    return compare_df, failed

def cal_merra2_str_adj_ghg(merra2, column, ghg_choice):
    fail = []
    all_rows = []
    
    # 1. Pre-index data for O(1) dictionary lookups
    col_map = column.set_index('Date').to_dict('index')
    # Pre-clean MERRA-2 and map by Date
    merra2_cleaned = merra2.drop_duplicates(subset=['Date'])
    m2_map = merra2_cleaned.set_index('Date').to_dict('index')
    
    unique_dates = column.Date.unique()

    for d in unique_dates:
        try:
            # Verify date exists in tower/aircraft column
            col_match = col_map.get(d)
            m2_row = m2_map.get(d)

            # Combine observations based on ghg_choice
            # 0 = CO2, Else = CH4
            if ghg_choice == 0:
                target_p = np.concatenate([np.atleast_1d(col_match['Pressure_2']), 
                                           np.atleast_1d(col_match['Pressure_1'])])
                target_ghg = np.concatenate([np.atleast_1d(col_match['CO2_2']), 
                                             np.atleast_1d(col_match['CO2_1'])])
                target_ghg_std = col_match['CO2_tower_std']
            else:                    
                target_p = np.concatenate([np.atleast_1d(col_match['Pressure_2']), 
                                           np.atleast_1d(col_match['Pressure_1'])])
                target_ghg = np.concatenate([np.atleast_1d(col_match['CH4_2']), 
                                             np.atleast_1d(col_match['CH4_1'])])
                target_ghg_std = col_match['CH4_tower_std']

            # 3. Get MERRA-2 profile
            total_p = m2_row['Pressure_boundary']
            partial_column = m2_row['Dry_air_partial_column']
            ps = m2_row.get('P0') # Surface pressure

            # 4. Clipping
            # Find closest index to 0.1 hPa using vectorized search
            total_p_arr = np.array(total_p)
            end_idx = np.abs(total_p_arr - 0.1).argmin()
            
            clip_total_p = total_p_arr[end_idx:]
            clip_partial_column = np.array(partial_column)[end_idx:]

            # 5. Binning
            binned_means, x, _ = binned_statistic(target_p, target_ghg, statistic='mean', bins=clip_total_p)
            
            # 6. Vectorized Fill NaNs (neighbors average)
            s = pd.Series(binned_means)
            neighbor_avg = pd.concat([s.shift(1), s.shift(-1)], axis=1).mean(axis=1)
            binned_means_filled = np.where(np.isnan(binned_means), neighbor_avg, binned_means)
            
            # 7. Weighted Average Calculation
            nan_mask = np.isnan(binned_means_filled)
            clean_ghg = binned_means_filled[~nan_mask]
            clean_weights = clip_partial_column[~nan_mask]
            
            if len(clean_weights) == 0:
                raise ValueError("All bins are NaN for this profile")

            merra2_ghg = np.sum(clean_ghg * clean_weights) / np.sum(clean_weights)
            
            # 8. Assembly
            bins_center = (x[:-1] + x[1:]) / 2
            
            all_rows.append({
                'Date': d, 'Year': d.year, 'Month': d.month, 'GHG_merra2': merra2_ghg, 
                'Pressure_column': target_p, 'GHG_column': target_ghg, 'GHG_tower_std': target_ghg_std,
                'Boundary': x, 'bins': bins_center, 'binned_ghg': binned_means, 
                'dry_air_partial_column': clip_partial_column
            })

        except Exception as e:
            fail.append([d, str(e)])
    
    # 9. Batch construction (much faster than repeated _append)
    if not all_rows:
        return pd.DataFrame(), fail
        
    out = pd.DataFrame(all_rows)
    return out.sort_values('Date').reset_index(drop=True), fail

## Detrending Functions

# create decimal year variable from np.datetime64
def to_decimal_year_64(dates, utc_hms):
    date_dt = pd.to_datetime(dates)
    specific_time = pd.to_timedelta(utc_hms)
    date_dt_full = date_dt + specific_time
    def _to_dec(d):
        start = datetime(d.year, 1, 1)
        end = datetime(d.year + 1, 1, 1)
        return d.year + ((d - start).total_seconds() / (end - start).total_seconds())
    return np.array([_to_dec(d) for d in date_dt_full])

# Get decimal year
def to_decimal_year(date_dt):
    date_obj = pd.to_datetime(date_dt)
    # Create start-of-year and start-of-next-year objects 
    year_start = type(date_obj)(date_obj.year, 1, 1)
    next_year_start = type(date_obj)(date_obj.year + 1, 1, 1)
    
    # Calculate durations in seconds
    year_duration = (next_year_start - year_start).total_seconds()
    time_elapsed = (date_obj - year_start).total_seconds()
    
    # Calculate decimal fraction
    return date_obj.year + (time_elapsed / year_duration)

# decimal to datetime
def decimal_to_datetime(decimal_date):
    year = int(decimal_date)
    remainder = decimal_date - year
    
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    year_duration = end_of_year - start_of_year
    
    return start_of_year + (remainder * year_duration)

    
def make_amp_df(l):
   out = pd.DataFrame({
        'Year': [l[i][0] for i in range(len(l))],
        'amp': [l[i][1] for i in range(len(l))],
        'peak_d': [decimal_to_datetime(l[i][2]).date() for i in range(len(l))],
        'peak': [l[i][3] for i in range(len(l))],
        'trough_d': [decimal_to_datetime(l[i][4]).date() for i in range(len(l))],
        'trough': [l[i][5] for i in range(len(l))]
        })
   return out
    
# Thoning's detrending (NOAA code)
def thoning_noaa(tccon, noon_hour):
    
    # 1. DATETIME CONVERSION & FILTERING
    time_series = pd.to_datetime(tccon['time'])
        
    mask_noon = time_series.hour.isin([noon_hour])
    noon_co2 = np.array(tccon['xco2_x2019'].values)[mask_noon]
    noon_ch4 = np.array(tccon['xch4'].values)[mask_noon]
        
    # 2. PREPARE AVERAGING DATAFRAME
    noon_data = pd.DataFrame({'Date': time_series[mask_noon].date,
                              'Date_dc': [to_decimal_year(d) for d in time_series[mask_noon]],
                              'co2': noon_co2,
                              'ch4': noon_ch4 * 1000 # Convert CH4 to ppb
                              })
        
    # 3. CALCULATE DAILY NOON AVERAGES
    daily_avg = noon_data.groupby('Date').agg({
        'co2': 'mean',
        'ch4': 'mean',
        'Date_dc': 'first'
        }).dropna().reset_index()
        
    t_obs = daily_avg['Date_dc'].values
    co2_obs = daily_avg['co2'].values
    ch4_obs = daily_avg['ch4'].values
    
    # Create even monthly grid at the middle of each month
    start_year = daily_avg['Date'].min().year
    end_year = daily_avg['Date'].max().year
    end_month = daily_avg['Date'].max().month
    
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-{end_month:02d}-30"
    
    grid_dates_ms = pd.date_range(start=start_str, end=end_str, freq='W')
    
    # 5. Shift every date forward by 14 days to lock them perfectly onto the 15th (Mid-Month)
    grid_dates = grid_dates_ms #+ pd.DateOffset(days=15)

    x0 = np.array([to_decimal_year(d) for d in grid_dates])
    x0_dt = [decimal_to_datetime(x).date() for x in x0]
    
    # CO2
    co2_filt = ccgfilt.ccgFilter(noon_data.Date_dc, noon_data.co2, shortterm=360, longterm=1080, numpolyterms=3, numharmonics=2)
    fit_co2 = co2_filt.getSmoothValue(x0)
    td_co2 = co2_filt.getTrendValue(x0)
    ssn_co2 = co2_filt.getHarmonicValue(x0)
    grw_co2 = np.insert(np.diff(td_co2) / np.diff(x0), 0, np.nan)
    
    # CH4
    ch4_filt = ccgfilt.ccgFilter(noon_data.Date_dc, noon_data.ch4, shortterm=360, longterm=1080, numpolyterms=3, numharmonics=2)
    fit_ch4 = ch4_filt.getSmoothValue(x0)
    td_ch4 = ch4_filt.getTrendValue(x0)
    ssn_ch4 = ch4_filt.getHarmonicValue(x0)
    grw_ch4 = np.insert(np.diff(td_ch4) / np.diff(x0), 0, np.nan)
    
    df = pd.DataFrame({
        'Date': x0_dt,
        'Year': [d.year for d in x0_dt],
        'Season': [get_season(d) for d in x0_dt],
        'fit_co2': fit_co2,
        'harmonics_co2': ssn_co2,
        'trend_co2': td_co2,
        'growth_co2': grw_co2,
        'fit_ch4': fit_ch4,
        'harmonics_ch4': ssn_ch4,
        'trend_ch4': td_ch4,
        'growth_ch4': grw_ch4
    })
    
    df_noon = pd.DataFrame({
        'Date': [decimal_to_datetime(x).date() for x in t_obs],
        'fit_co2': co2_filt.getSmoothValue(t_obs),
        'harmonics_co2': co2_filt.getHarmonicValue(t_obs),
        'trend_co2': co2_filt.getTrendValue(t_obs),
        'fit_ch4': ch4_filt.getSmoothValue(t_obs),
        'harmonics_ch4': ch4_filt.getHarmonicValue(t_obs),
        'trend_ch4': ch4_filt.getTrendValue(t_obs),
    })
    
    noon_fit = pd.merge(daily_avg, df_noon, on='Date', how='inner')
    noon_fit['detrend_co2'] = noon_fit.co2 - noon_fit.trend_co2
    noon_fit['detrend_ch4'] = noon_fit.ch4 - noon_fit.trend_ch4
        
    co2_amp = make_amp_df(co2_filt.getAmplitudes())
    ch4_amp = make_amp_df(ch4_filt.getAmplitudes())
    
    return df, noon_fit, co2_amp, ch4_amp

def thoning_noaa_local(df):
    
    t_obs = np.array([to_decimal_year(d) for d in df.Date])
    ghg_obs = np.array(df.GHG_merra2.values)

    ghg_filt = ccgfilt.ccgFilter(t_obs, ghg_obs, shortterm=360, longterm=1080, numpolyterms=3, numharmonics=2)
    
    # Create even monthly grid at the middle of each month
    start_year = df['Date'].min().year
    end_year = df['Date'].max().year
    end_month = df['Date'].max().month
    
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-{end_month:02d}-30"
    
    grid_dates_ms = pd.date_range(start=start_str, end=end_str, freq='W')
    
    # 5. Shift every date forward by 14 days to lock them perfectly onto the 15th (Mid-Month)
    grid_dates = grid_dates_ms

    x0 = np.array([to_decimal_year(d) for d in grid_dates])
    x0_dt = [decimal_to_datetime(x).date() for x in x0]
    fit_ghg = ghg_filt.getSmoothValue(x0)
    td_ghg = ghg_filt.getTrendValue(x0)
    ssn_ghg = ghg_filt.getHarmonicValue(x0)
    grw_ghg = np.insert(np.diff(td_ghg) / np.diff(x0), 0, np.nan)
    
    df_int = pd.DataFrame({
        'Date': x0_dt,
        'Year': [d.year for d in x0_dt],
        'Month': [d.month for d in x0_dt],
        'Season': [get_season(d) for d in x0_dt],
        'fit': fit_ghg,
        'harmonics': ssn_ghg,
        'trend': td_ghg,
        'growth': grw_ghg
    })
    
    df_amp = make_amp_df(ghg_filt.getAmplitudes())
    
    df_fit = pd.DataFrame({
        'Date': df.Date,
        'fit': ghg_filt.getSmoothValue(t_obs),
        'harmonics': ghg_filt.getHarmonicValue(t_obs),
        'trend': ghg_filt.getTrendValue(t_obs)
        })
    
    long_fit = pd.merge(df, df_fit, on=['Date'], how='inner')
    long_fit['detrend'] = long_fit.GHG_merra2 - long_fit.trend
    
    return df_int, long_fit, df_amp


## Plotting functions
def add_sigma_to_fill(df, sigma_unctower, sigma_notower):
    df['sigma'] = ''    
    df['sigma'] = np.where(df.filter(regex='tower_std').isna(),
                           sigma_notower,
                           sigma_unctower)
    return df

def get_alpha_col(color, alpha=0.5):
    return mcolors.to_rgba(color, alpha=alpha)

def detect_outliers_iqr(df, diff_col, sig_col, multiplier=1.5):
    
    df = df.copy()
    
    # Calculate the Normalized Difference (Z-score equivalent)
    df['norm_diff'] = df[diff_col] / df[sig_col]
    
    # Calculate the 25th (Q1) and 75th (Q3) percentiles of the NORMALIZED column
    Q1 = df['norm_diff'].quantile(0.25)
    Q3 = df['norm_diff'].quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the upper and lower fences (thresholds)
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    # Filter the data based on the fences
    anomalies = df[(df['norm_diff'] < lower_bound) | (df['norm_diff'] > upper_bound)]
    normal_data = df[(df['norm_diff'] >= lower_bound) & (df['norm_diff'] <= upper_bound)]
    
    # Optional: Print a summary
    print(f"--- Uncertainty-Normalized IQR Detection for '{diff_col}' ---")
    print(f"Q1: {Q1:.3f} | Q3: {Q3:.3f} | IQR: {IQR:.3f}")
    print(f"Lower Fence: {lower_bound:.3f}")
    print(f"Upper Fence: {upper_bound:.3f}")
    print(f"Total Normal Points: {len(normal_data)}")
    print(f"Total Anomalies Found: {len(anomalies)}\n")
    
    return anomalies, normal_data, (lower_bound, upper_bound)


def calculate_seasonal_weighted_stats(df, residual_col, uncert_col, season_col):
    results = []
    
    for season, group in df.groupby(season_col):
        residuals = group[residual_col].values
        uncertainties = group[uncert_col].values
        N = len(residuals)
        
        # Guard: If a season has no data
        if N == 0:
            continue
            
        # 1. Calculate raw Inverse-Variance Weights
        raw_weights = 1.0 / (uncertainties ** 2)
        
        # 2. Normalize weights so their sum equals N
        # This prevents the sum_weights < 1 error in statsmodels
        normalized_weights = raw_weights * (N / np.sum(raw_weights))
        
        # 3. Feed into statsmodels
        weighted_stats = DescrStatsW(residuals, weights=normalized_weights, ddof=1)
        
        weighted_mean = weighted_stats.mean
        
        # Guard: Standard Error requires at least 2 observations
        if N > 1:
            weighted_se = weighted_stats.std_mean
        else:
            weighted_se = np.nan # Undefined for a single point
            
        results.append({
            'Season': season,
            'N_Observations': N,
            'Weighted_Mean_Bias': weighted_mean,
            'Weighted_Standard_Error': weighted_se
        })
        
    return pd.DataFrame(results)


def compare_tg(gosat, tccon, gas):
    test = pd.merge(gosat, tccon[['Date', gas]], on='Date', how='inner')
    test['Diff'] = test[f'x{gas}_gosat'] - test[gas]
    test = test.dropna(subset=['Diff'])
    test['b'] = abs(test.Diff)/test[f'x{gas}_gosat']*100
    
    # Aggregate in a single pass to eliminate redundant grouping calculations
    stats = test.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
    se = stats['std'] / np.sqrt(stats['count'])
    # b_mean = test.groupby('Season')['b'].agg('mean')
    
    return stats['mean'].tolist(), se.tolist(), stats.index.tolist(), \
        stats['count'].tolist(), test.b.mean(), test.Diff.values

def compare_tg_df(gosat, tccon, gas):     
    test = pd.merge(gosat, tccon[['Date', gas]], on='Date', how='inner')
    test['Diff'] = test[f'x{gas}_gosat'] - test[gas]
    test = test.dropna(subset=['Diff'])
    test["SA_NIR"] = test["SA_NIR"].mask(test["SA_NIR"] <0)
        
    return test

# --- Generalized Plotting Helpers ---

def plot_timeseries_panel(ax, gosat, cak_norm, cak_out, m2_norm, m2_anom,
                          trend, site_name, label_id, gas_info):
    col = gas_info['col_key']
    unit = gas_info['unit']
    gas = gas_info['name']
    
    if len(cak_out) == 0 | len(m2_anom) == 0:
        pass
    # 0. GOSAT all    
    ax.scatter(gosat.Date, gosat[gas_info['gas_key']], marker='o', s=1.2, c='wheat', alpha=.2,
               label=rf'$X_{{{gas}, all}}^{{GOSAT}}$')
    
    # 1. MERRA-2 Data
    ax.errorbar(m2_norm.Date, m2_norm.GHG_merra2, yerr=m2_norm.sigma, fmt='^', markersize=2.5, 
                elinewidth=.4, capsize=0, color='skyblue', markeredgecolor='steelblue', 
                ecolor=get_alpha_col('steelblue'), markeredgewidth=.5, label=rf'$X_{{{gas}}}^{{ind}}$')
    # 2. CAK Data
    for data, color, edge, lab in [(cak_norm, 'mediumpurple', 'rebeccapurple', rf'$X_{{{gas}}}^{{CAK}}$'),
                                   (cak_out, 'mediumpurple', 'crimson', rf'$X_{{{gas}, error}}^{{CAK}}$')]:
        if data.empty: continue
        agg = data.groupby('Date').agg({col: 'mean', 'sigma': 'mean'}).reset_index()
        ax.errorbar(agg.Date, agg[col], yerr=agg.sigma, fmt='D', markersize=2, 
                    elinewidth=.4, capsize=0, color=color, markeredgecolor=edge, 
                    ecolor=get_alpha_col(edge), markeredgewidth=.5, label=lab)

    
    ax.errorbar(m2_anom.Date, m2_anom.GHG_merra2, yerr=m2_anom.sigma, fmt='^', markersize=2.5, 
                elinewidth=.4, capsize=0, color='skyblue', markeredgecolor='brown', 
                ecolor=get_alpha_col('brown'), markeredgewidth=.5, label=rf'$X_{{{gas}, outlier}}^{{ind}}$', zorder=4)

    # 3. Fitting & GOSAT matched
    ax.plot(trend.Date, trend.filter(regex='fit'), linewidth=.4, color='grey', label='Fitting Curve', zorder=1)
    
    gosat_agg = pd.concat([cak_norm, cak_out]).groupby('Date')[gas_info['gosat_key']].mean()
    ax.scatter(gosat_agg.index, gosat_agg.values, marker='o', s=2.8, c='khaki', edgecolors='black',
               linewidths=.4, label=rf'$X_{{{gas}, matched}}^{{GOSAT}}$')

    # 4. Texts
    ax.text(0.98, 0.96, label_id, transform=ax.transAxes, fontsize=10, ha='right', va='top')
    ax.text(0.02, 0.96, site_name, transform=ax.transAxes, fontsize=10, ha='left', va='top')
    ax.set_ylabel(rf'X{gas} [{unit}]')

def plot_seasonal_panel(ax, cak_norm, m2_reg, gosat_raw, label_id, gas_info, text_y1, text_y2):
    ax.axhline(y=0, color='black', linestyle='--', linewidth=.5)
    gas = gas_info['name']
    unit = gas_info['unit']
    
    # 1. CAK Diff
    agg = calculate_seasonal_weighted_stats(cak_norm, 'Diff', 'sigma', 'Season')
    r = agg.Season.unique()
    
    ax.errorbar(r - 0.15, agg['Weighted_Mean_Bias'], yerr=agg['Weighted_Standard_Error'],
                fmt='D', markersize=4, capsize=2, elinewidth=1,
                color='mediumpurple', ecolor='rebeccapurple', markeredgecolor="rebeccapurple",
                label=rf'$X_{{{gas}}}^{{GOSAT}}-X_{{{gas}}}^{{CAK}}$')
    
    for i, season in enumerate(r):
        ax.text(season-0.2, text_y1, f'n={agg.iloc[i]["N_Observations"]:.0f}', fontsize=5, color='rebeccapurple')

    # 2. MERRA-2 Diff
    m2_merged = pd.merge(m2_reg, cak_norm[['Date', gas_info['gosat_key']]], on='Date', how='inner')
    m2_merged['Diff'] = m2_merged[gas_info['gosat_key']] - m2_merged['GHG_merra2']
    
    m2_agg = calculate_seasonal_weighted_stats(m2_merged, 'Diff', 'sigma', 'Season')
    r_m2 = m2_agg.Season.unique()
    
    ax.errorbar(r_m2 + 0.15, m2_agg['Weighted_Mean_Bias'], yerr=m2_agg['Weighted_Standard_Error'],
                fmt='^', markersize=5, capsize=2, elinewidth=1,
                color='skyblue', ecolor='steelblue', markeredgecolor="steelblue",
                label=rf'$X_{{{gas}}}^{{GOSAT}}-X_{{{gas}}}^{{ind}}$')
    
    for i, season in enumerate(r_m2):
        ax.text(season-0.2, text_y2, f'n={m2_agg.iloc[i]["N_Observations"]:.0f}', fontsize=5, color='steelblue')

    ax.text(0.98, 0.96, label_id, transform=ax.transAxes, fontsize=10, ha='right', va='top')
    ax.set_ylabel(rf'$\Delta$X{gas} [{unit}]')

## Confidence ellipse
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Utility to calculate and add a covariance ellipse to an axes object.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Eigenvalue calculation for radius
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, linewidth=.4, **kwargs)

    # Standard deviation scaling
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)