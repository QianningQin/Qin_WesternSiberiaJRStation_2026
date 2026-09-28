#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Analysis
"""
# %% Import modules and functions

# import packages
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
import glob
from datetime import datetime

# load functions in the same directory
import readin as rd
import analysis as ana

# %% Define file paths
# Change /Users/<qnqin>/ to your user name in /Users/<username>

# GOSAT L2 total columns
dir_gosat_co2 = "/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/GOSAT/L2_CO2_column_amount/"
dir_gosat_ch4 = "/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/GOSAT/L2_CH4_column_amount/"

# JR-Station & CRDS
JRStation_file_path = r"/Users/qnqin/Library/CloudStorage/Box-Box/Common_Data/Data/JR-STATION"
crds_file_path = "/Users/qnqin/Library/CloudStorage/Box-Box/Common_Data/Data/JR-STATION/CRDS/DEM"
ch4_files = glob.glob(f'{crds_file_path}/ch4*.txt')

# Aircraft
dir_aircraft = r'/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/Aircraft/'

# MERRA-2 boundary layers data
merra2_tp_pbl_file_path = "/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/MERRA-2_tp_pbl"

# MERRA-2 air density variables
merra2_ad_path = "/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/MERRA-2_air_density/"

# CAMSinv CO2 and CH4 profile
camsinv_co2_path = r"/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/ECMWF_CO2/"
camsinv_ch4_path = r"/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/ECMWF_CH4/"

# SCISAT stratospheric reference data
scisat_path = r"/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/SciSat/NETCDF_mol_t_split/"

# IGRA altitude-pressure reference data
igra_nov_path = '/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/IGRA/NOV/RSM00029634-data.txt'
igra_dem_path = '/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/IGRA/NOY/RSM00023933-data.txt'

# TCCON data
tccon_dir = r'/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Data_Qin/TCCON/'

# %% Load GOSAT Data
# GOSAT XCO2
GOSAT_years = range(2009,2023)

# JR-Station BRZ
box_brz = [53.5, 58.5, 81.8, 87.3]
gosat_co2_brz = rd.read_in_gosat_co2(dir_gosat_co2, box_brz, GOSAT_years)
# JR-Station DEM
box_dem = [56.8, 62.8, 67.9, 73.9]
gosat_co2_dem = rd.read_in_gosat_co2(dir_gosat_co2, box_dem, GOSAT_years)
# TCCON sites
box_so = [64.9, 69.9, 24.1, 29.1]
gosat_co2_so = rd.read_in_gosat_co2(dir_gosat_co2, box_so, GOSAT_years)
box_es = [51.8, 56.8, -107.5, -102.5]
gosat_co2_es = rd.read_in_gosat_co2(dir_gosat_co2, box_es, GOSAT_years)
box_bi = [49.7, 55.7, 20.5, 25.5]
gosat_co2_bi = rd.read_in_gosat_co2(dir_gosat_co2, box_bi, GOSAT_years)
box_br = [49.6, 55.6, 6.3, 11.3]
gosat_co2_br = rd.read_in_gosat_co2(dir_gosat_co2, box_br, GOSAT_years)
box_pa = [43.4, 48.4, -92.8, -87.8]
gosat_co2_pa = rd.read_in_gosat_co2(dir_gosat_co2, box_pa, GOSAT_years)

# GOSAT XCH4
gosat_ch4_brz = rd.read_in_gosat_ch4(dir_gosat_ch4, box_brz, GOSAT_years)
gosat_ch4_dem = rd.read_in_gosat_ch4(dir_gosat_ch4, box_dem, GOSAT_years)
gosat_ch4_so = rd.read_in_gosat_ch4(dir_gosat_ch4, box_so, GOSAT_years)
gosat_ch4_bi = rd.read_in_gosat_ch4(dir_gosat_ch4, box_bi, GOSAT_years)
gosat_ch4_br = rd.read_in_gosat_ch4(dir_gosat_ch4, box_br, GOSAT_years)
gosat_ch4_es = rd.read_in_gosat_ch4(dir_gosat_ch4, box_es, GOSAT_years)
gosat_ch4_pa = rd.read_in_gosat_ch4(dir_gosat_ch4, box_pa, GOSAT_years)

# Add distance to GOSAT observations
brz_coord = (56, 84.3)
gosat_co2_brz = ana.get_distance(gosat_co2_brz, brz_coord)
gosat_ch4_brz = ana.get_distance(gosat_ch4_brz, brz_coord)
dem_coord = (59.8, 70.9)
gosat_co2_dem = ana.get_distance(gosat_co2_dem, dem_coord)
gosat_ch4_dem = ana.get_distance(gosat_ch4_dem, dem_coord)

del dir_gosat_co2, dir_gosat_ch4, brz_coord, dem_coord

# Run the following code in console to delete unnecesssary variables
# %reset_selective -f ^box

# %% Load tower Data

tower_brz = rd.read_in_tower(JRStation_file_path, 'BRZ*.txt', 140)
tower_dem = rd.read_in_tower(JRStation_file_path, 'DEM*.txt', 132)

# Replace NDIR with CRDS CH4 at DEM tower
for i in range(len(ch4_files)):
    with open(ch4_files[i], 'r', encoding='utf-8') as file:
        f = pd.read_csv(file, sep=' ').drop('EPOCH_TIME', axis=1).rename(columns={'DATE': 'Date', 'TIME': 'Time', 'CH4_94scale': 'CH4'})
        f.CH4 = f.CH4 * 1000
        # separate hour and minute info from Time column
        f['Date'] = [datetime.strptime(x, '%Y-%m-%d').date() for x in f.Date.values]
        f['Hour'] = [datetime.strptime(x,'%H:%M:%S').hour for x in f.Time.values]
        f['Minute'] = [datetime.strptime(x,'%H:%M:%S').minute for x in f.Time.values]
        
        # separate CRDS to high and low inlet, replace original cell with closest time
        for d in f.Date.unique():
            f_this_d = f.loc[f.Date == d]
            f_this_d_hr = f_this_d.Hour.unique().tolist()
                
            f_this_d_high = f_this_d.loc[f_this_d.valve == 1]
            f_this_d_low = f_this_d.loc[f_this_d.valve == 2]
                
            tower_this_d = tower_dem.loc[tower_dem.Date == d]
                
            # for hours with CRDS obs., replace original obs. with closest CRDS obs. (at minutes)
            for hr in f_this_d_hr:
                tower_this_d_hr_mnt = tower_this_d.loc[tower_this_d.Hour == hr].MINUTE.values
                f_this_d_hr_mnt = f_this_d.loc[f_this_d.Hour == hr].Minute.unique().tolist()
                for mnt in tower_this_d_hr_mnt:
                    closest_mnt = ana.findClosestValue(f_this_d_hr_mnt, mnt)
                    f_this_mnt_ch4_high = f_this_d_high.loc[(f_this_d_high.Hour == hr) & (f_this_d_high.Minute == closest_mnt)].CH4.mean()
                    f_this_mnt_ch4_low = f_this_d_low.loc[(f_this_d_low.Hour == hr) & (f_this_d_low.Minute == closest_mnt)].CH4.mean()
                    
                    idx = tower_dem.loc[(tower_dem.Date == d) & (tower_dem.Hour == hr) & (tower_dem.MINUTE == mnt)].index.item()
                    inlet_height = tower_dem.at[idx, 'Vp']
                    if inlet_height == 63:
                        tower_dem.at[idx, 'CH4'] = f_this_mnt_ch4_high
                    else:
                        tower_dem.at[idx, 'CH4'] = f_this_mnt_ch4_low

# Delete paths and unnecesssary variables
del JRStation_file_path, crds_file_path, ch4_files, mnt, hr, i, idx, inlet_height, d, closest_mnt

# Run the following code in console 
# %reset_selective -f ^f
# %reset_selective -f ^tower_this_d
# %% Load aircraft Data
# SUR (utc +5)
aircraft_sur = pd.read_csv(dir_aircraft+'SUR_all_year_TU_NIES_corr.csv')
aircraft_sur, aircraft_dates_sur = rd.read_in_aircraft(aircraft_sur, "Etc/GMT+0", '%Y-%m-%d %H:%M:%S', 1, 1, 1)

# NOV (utc +5)
aircraft_nov = pd.read_csv(dir_aircraft+'NOV_all_year_NIES_corr.csv')
aircraft_nov, aircraft_dates_nov = rd.read_in_aircraft(aircraft_nov, "Etc/GMT+0", '%Y-%m-%d %H:%M:%S', 1, 1, 1)

del dir_aircraft

# %% Load MERRA-2 TP, PBL, P0, Temp

# UTC +6
merra2_tp_pbl_brz, brz_failed = rd.get_merra2_tp_pbl(merra2_tp_pbl_file_path, 56, 84.375, 7)
# UTC +5
merra2_tp_pbl_dem, dem_failed = rd.get_merra2_tp_pbl(merra2_tp_pbl_file_path, 60, 70.625, 8)

# %% Load CAMSinv data
camsinv_ch4_brz = rd.get_camsinv_ch4(camsinv_ch4_path, 55.5, 84.5)
camsinv_ch4_dem = rd.get_camsinv_ch4(camsinv_ch4_path, 59.5, 70.5)

camsinv_co2_brz = rd.get_camsinv_co2(camsinv_co2_path, 55.5, 84.5, 7)
camsinv_co2_dem = rd.get_camsinv_co2(camsinv_co2_path, 59.5, 70.5, 8)
camsinv_co2_so = rd.get_camsinv_co2(camsinv_co2_path, 67.4, 26.6, 9)

del camsinv_co2_path, camsinv_ch4_path

# %% Load SCISAT data

scisat_brz, failed = rd.read_in_scisat(scisat_path, camsinv_ch4_brz.Date.unique(), 56, 84.5)
scisat_dem, failed = rd.read_in_scisat(scisat_path, camsinv_ch4_dem.Date.unique(), 60, 70.5)
scisat_so, failed = rd.read_in_scisat(scisat_path, camsinv_co2_so.Date.unique(), 67, 26.5)

del scisat_path

# %% Filter GOSAT to aircraft dates (± 1 day)

gosat_co2_dates_brz = ana.get_aircraft_gosat_dates(aircraft_dates_nov, gosat_co2_brz.Date.unique())  # 30 days
gosat_co2_dates_dem = ana.get_aircraft_gosat_dates(aircraft_dates_sur, gosat_co2_dem.Date.unique())  # 36 days
gosat_ch4_dates_brz = ana.get_aircraft_gosat_dates(aircraft_dates_nov, gosat_ch4_brz.Date.unique())  # 26 days
gosat_ch4_dates_dem = ana.get_aircraft_gosat_dates(aircraft_dates_sur, gosat_ch4_dem.Date.unique())  # 36 days

gosat_co2_brz_filtered = gosat_co2_brz.loc[gosat_co2_brz.Date.isin(gosat_co2_dates_brz)].reset_index(drop=True)  # 128
gosat_co2_dem_filtered = gosat_co2_dem.loc[gosat_co2_dem.Date.isin(gosat_co2_dates_dem)].reset_index(drop=True)  # 120

gosat_ch4_brz_filtered = gosat_ch4_brz.loc[gosat_ch4_brz.Date.isin(gosat_ch4_dates_brz)].reset_index(drop=True)  # 114
gosat_ch4_dem_filtered = gosat_ch4_dem.loc[gosat_ch4_dem.Date.isin(gosat_ch4_dates_dem)].reset_index(drop=True)  # 120

# %% Filter tower to aircraft dates (± 1 day) and get afternoon mean, std.

tower_brz_compare = tower_brz.loc[tower_brz.Date.isin(gosat_co2_dates_brz)].reset_index(drop=True)
tower_dem_compare = tower_dem.loc[tower_dem.Date.isin(gosat_co2_dates_dem)].reset_index(drop=True)
tower_brz_long = tower_brz.loc[tower_brz.Date.isin(aircraft_dates_nov)].reset_index(drop=True)
tower_dem_long = tower_dem.loc[tower_dem.Date.isin(aircraft_dates_sur)].reset_index(drop=True)

tower_brz_compare_afternoon, failed = ana.tower_filter(tower_brz_compare, 7, 12)
tower_dem_compare_afternoon, failed = ana.tower_filter(tower_dem_compare, 8, 13)
tower_brz_long_afternoon, failed = ana.tower_filter(tower_brz_long, 7, 12)
tower_dem_long_afternoon, failed = ana.tower_filter(tower_dem_long, 8, 13)

# %% Load IGRA

# Novosibirsk: 142 m a.s.l, GMT+6
igra_nov = rd.read_in_igra(igra_nov_path, 0)
igra_nov.Altitude = (igra_nov.Altitude - 142)/1000

#HANTY-MANSIJSK (Surgut): 40 a.s.l, GMT+5
igra_dem = rd.read_in_igra(igra_dem_path, 0)
igra_dem.Altitude = (igra_dem.Altitude - 40)/1000

del igra_nov_path, igra_dem_path

# %% Altitude-Pressure transBarometric curve fitting

P0 = 1013
igra_param_dem = ana.curve_fitting(igra_dem)
igra_param_nov = ana.curve_fitting(igra_nov)

# %% Construct discrete profile

column_compare_co2_dem, failed = ana.make_more_column(aircraft_sur, tower_dem_compare_afternoon,
                                                      igra_param_dem, merra2_tp_pbl_dem, gosat_co2_dates_dem, 0)
column_compare_co2_brz, failed = ana.make_more_column(aircraft_nov, tower_brz_compare_afternoon,
                                                      igra_param_nov, merra2_tp_pbl_brz, gosat_co2_dates_brz, 0)

column_compare_ch4_dem, failed = ana.make_more_column(aircraft_sur, tower_dem_compare_afternoon,
                                                      igra_param_dem, merra2_tp_pbl_dem, gosat_ch4_dates_dem, 1)
column_compare_ch4_brz, failed = ana.make_more_column(aircraft_nov, tower_brz_compare_afternoon,
                                                      igra_param_nov, merra2_tp_pbl_brz, gosat_ch4_dates_brz, 1)

column_long_co2_dem, failed = ana.make_more_column(aircraft_sur, tower_dem_long_afternoon,
                                                   igra_param_dem, merra2_tp_pbl_dem, aircraft_dates_sur, 0)
column_long_co2_brz, failed = ana.make_more_column(aircraft_nov, tower_brz_long_afternoon,
                                                   igra_param_nov, merra2_tp_pbl_brz, aircraft_dates_nov, 0)

column_long_ch4_dem, failed = ana.make_more_column(aircraft_sur, tower_dem_long_afternoon,
                                                   igra_param_dem, merra2_tp_pbl_dem, aircraft_dates_sur, 1)
column_long_ch4_brz, failed = ana.make_more_column(aircraft_nov, tower_brz_long_afternoon,
                                                   igra_param_nov, merra2_tp_pbl_brz, aircraft_dates_nov, 1)

column_long_co_dem, failed = ana.make_more_column(aircraft_sur, tower_dem_long_afternoon,
                                                  igra_param_dem, merra2_tp_pbl_dem, aircraft_dates_sur, 2)
column_long_co_brz, failed = ana.make_more_column(aircraft_nov, tower_brz_long_afternoon,
                                                  igra_param_nov, merra2_tp_pbl_brz, aircraft_dates_nov, 2)

# %% Combine MERRA-2 PBL and TP with param and fitted results

column_tofit_compare_co2_brz = pd.merge(merra2_tp_pbl_brz[['Date','PBL_Pressure','TP_B']], column_compare_co2_brz,
                                        on="Date").sort_values(by='Date').reset_index(drop=True)
column_tofit_compare_co2_dem = pd.merge(merra2_tp_pbl_dem[['Date','PBL_Pressure','TP_B']], column_compare_co2_dem,
                                        on="Date").sort_values(by='Date').reset_index(drop=True)

column_tofit_compare_ch4_brz = pd.merge(merra2_tp_pbl_brz[['Date','PBL_Pressure','TP_B']], column_compare_ch4_brz,
                                        on="Date").sort_values(by='Date').reset_index(drop=True)
column_tofit_compare_ch4_dem = pd.merge(merra2_tp_pbl_dem[['Date','PBL_Pressure','TP_B']], column_compare_ch4_dem,
                                        on="Date").sort_values(by='Date').reset_index(drop=True)

column_tofit_long_co2_brz = pd.merge(merra2_tp_pbl_brz[['Date','PBL_Pressure','TP_B']], column_long_co2_brz,
                                     on="Date").sort_values(by='Date').reset_index(drop=True)
column_tofit_long_co2_dem = pd.merge(merra2_tp_pbl_dem[['Date','PBL_Pressure','TP_B']], column_long_co2_dem,
                                     on="Date").sort_values(by='Date').reset_index(drop=True)

column_tofit_long_ch4_brz = pd.merge(merra2_tp_pbl_brz[['Date','PBL_Pressure','TP_B']], column_long_ch4_brz,
                                     on="Date").sort_values(by='Date').reset_index(drop=True)
column_tofit_long_ch4_dem = pd.merge(merra2_tp_pbl_dem[['Date','PBL_Pressure','TP_B']], column_long_ch4_dem,
                                     on="Date").sort_values(by='Date').reset_index(drop=True)

column_tofit_long_co_dem = pd.merge(merra2_tp_pbl_dem[['Date','PBL_Pressure','TP_B']], column_long_co_dem,
                                    on="Date").sort_values(by='Date').reset_index(drop=True)
column_tofit_long_co_brz = pd.merge(merra2_tp_pbl_brz[['Date','PBL_Pressure','TP_B']], column_long_co_brz,
                                    on="Date").sort_values(by='Date').reset_index(drop=True)

# %% Construct continuous profile

column_compare_ch4_brz, failed_brz = ana.CH4_column_generator(column_tofit_compare_ch4_brz, camsinv_ch4_brz)
column_compare_ch4_dem, failed_dem = ana.CH4_column_generator(column_tofit_compare_ch4_dem, camsinv_ch4_dem)

column_long_ch4_brz, failed_brz = ana.CH4_column_generator(column_tofit_long_ch4_brz, camsinv_ch4_brz)
column_long_ch4_dem, failed_dem = ana.CH4_column_generator(column_tofit_long_ch4_dem, camsinv_ch4_dem)

column_compare_co2_brz, failed_brz = ana.CO2_column_generator(column_tofit_compare_co2_brz, camsinv_co2_brz)
column_compare_co2_dem, failed_dem = ana.CO2_column_generator(column_tofit_compare_co2_dem, camsinv_co2_dem)

column_long_co2_brz, failed_brz = ana.CO2_column_generator(column_tofit_long_co2_brz, camsinv_co2_brz)
column_long_co2_dem, failed_dem = ana.CO2_column_generator(column_tofit_long_co2_dem, camsinv_co2_dem)

column_long_co_dem, failed_dem = ana.CO_PBL_TP_generator(column_tofit_long_co_dem)
column_long_co_brz, failed_brz = ana.CO_PBL_TP_generator(column_tofit_long_co_brz)

# %% Load/Calculate MERRA-2 dry air mass

merra2_dry_air_mass_brz, failed = rd.get_MERRA2_air_mass(merra2_ad_path, 
                                                         7, 55.5, 84.375)
merra2_dry_air_mass_dem, failed = rd.get_MERRA2_air_mass(merra2_ad_path,
                                                         8, 60, 70.625)
del merra2_ad_path

# %% Total column calculation (independent)

# MERRA-2 XCO2
long_co2_brz, failed_brz = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_brz, 0)
long_co2_dem, failed_dem = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_dem, 0)

# MERRA-2 XCH4
long_ch4_brz, failed_brz = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_brz, 1)
long_ch4_dem, failed_dem = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_dem, 1)

# %% Uncertainty 1/5: TP perturbation

# Estimate perturbtion bounds        
merra2_tp_cal_nov, failed = rd.get_merra2_tp_pbl(merra2_tp_pbl_file_path, 55, 83.125, 12)
merra2_tp_cal_sur, failed = rd.get_merra2_tp_pbl(merra2_tp_pbl_file_path, 61, 69.375, 12)

tp_cal_nov, tp_reg_nov = ana.get_calibrated_tp(igra_param_nov, merra2_tp_cal_nov)
tp_cal_sur, tp_reg_sur = ana.get_calibrated_tp(igra_param_dem, merra2_tp_cal_sur)

del merra2_tp_pbl_file_path

# %% Tropopause perturbation analysis (CH4)

# Adjust TP in profile
tp_cal_adj_brz = ana.get_tp_seasonal_var(tp_reg_nov)
column_compare_ch4_tp_adj1_brz, failed = ana.CH4_column_tp_adj(column_tofit_compare_ch4_brz, camsinv_ch4_brz,
                                                               tp_cal_adj_brz, 1)
column_compare_ch4_tp_adj0_brz, failed = ana.CH4_column_tp_adj(column_tofit_compare_ch4_brz, camsinv_ch4_brz,
                                                               tp_cal_adj_brz, 0)
tp_cal_adj_dem = ana.get_tp_seasonal_var(tp_reg_sur)
column_compare_ch4_tp_adj1_dem, failed = ana.CH4_column_tp_adj(column_tofit_compare_ch4_dem, camsinv_ch4_dem,
                                                               tp_cal_adj_dem, 1)
column_compare_ch4_tp_adj0_dem, failed = ana.CH4_column_tp_adj(column_tofit_compare_ch4_dem, camsinv_ch4_dem,
                                                               tp_cal_adj_dem, 0)

column_long_ch4_tp_adj1_brz, failed = ana.CH4_column_tp_adj(column_tofit_long_ch4_brz, camsinv_ch4_brz,
                                                            tp_cal_adj_brz, 1)
column_long_ch4_tp_adj0_brz, failed = ana.CH4_column_tp_adj(column_tofit_long_ch4_brz, camsinv_ch4_brz,
                                                            tp_cal_adj_brz, 0)
column_long_ch4_tp_adj1_dem, failed = ana.CH4_column_tp_adj(column_tofit_long_ch4_dem, camsinv_ch4_dem,
                                                            tp_cal_adj_dem, 1)
column_long_ch4_tp_adj0_dem, failed = ana.CH4_column_tp_adj(column_tofit_long_ch4_dem, camsinv_ch4_dem,
                                                            tp_cal_adj_dem, 0)

# Calculation
compare_gosat_ch4_tpadj1_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tp_adj1_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_tpadj0_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tp_adj0_brz, gosat_ch4_brz_filtered)

compare_gosat_ch4_tpadj1_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tp_adj1_dem, gosat_ch4_dem_filtered)
compare_gosat_ch4_tpadj0_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tp_adj0_dem, gosat_ch4_dem_filtered)

long_ch4_tpadj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_tp_adj0_brz, 1)
long_ch4_tpadj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_tp_adj1_brz, 1)

long_ch4_tpadj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_tp_adj0_dem, 1)
long_ch4_tpadj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_tp_adj1_dem, 1)

# %% Tropopause perturbation analysis (CO2)

# Adjust TP in profile
column_long_co2_tp_adj1_brz, failed = ana.CO2_column_tp_adj(column_tofit_long_co2_brz, camsinv_co2_brz,
                                                            tp_cal_adj_brz, 1)
column_long_co2_tp_adj0_brz, failed = ana.CO2_column_tp_adj(column_tofit_long_co2_brz, camsinv_co2_brz,
                                                            tp_cal_adj_brz, 0)
column_compare_co2_tp_adj1_brz, failed = ana.CO2_column_tp_adj(column_tofit_compare_co2_brz, camsinv_co2_brz,
                                                               tp_cal_adj_brz, 1)
column_compare_co2_tp_adj0_brz, failed = ana.CO2_column_tp_adj(column_tofit_compare_co2_brz, camsinv_co2_brz,
                                                               tp_cal_adj_brz, 0)

column_long_co2_tp_adj1_dem, failed = ana.CO2_column_tp_adj(column_tofit_long_co2_dem, camsinv_co2_dem,
                                                            tp_cal_adj_dem, 1)
column_long_co2_tp_adj0_dem, failed = ana.CO2_column_tp_adj(column_tofit_long_co2_dem, camsinv_co2_dem,
                                                            tp_cal_adj_dem, 0)
column_compare_co2_tp_adj1_dem, failed = ana.CO2_column_tp_adj(column_tofit_compare_co2_dem, camsinv_co2_dem,
                                                               tp_cal_adj_dem, 1)
column_compare_co2_tp_adj0_dem, failed = ana.CO2_column_tp_adj(column_tofit_compare_co2_dem, camsinv_co2_dem,
                                                               tp_cal_adj_dem, 0)

# Calculation
compare_gosat_co2_tpadj1_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_tp_adj1_brz, gosat_co2_brz_filtered)
compare_gosat_co2_tpadj0_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_tp_adj0_brz, gosat_co2_brz_filtered)

compare_gosat_co2_tpadj1_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_tp_adj1_dem, gosat_co2_dem_filtered)
compare_gosat_co2_tpadj0_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_tp_adj0_dem, gosat_co2_dem_filtered)

long_co2_tpadj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_tp_adj0_brz, 0)
long_co2_tpadj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_tp_adj1_brz, 0)

long_co2_tpadj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_tp_adj0_dem, 0)
long_co2_tpadj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_tp_adj1_dem, 0)

# %% Total column calculation (CAK)

compare_gosat_ch4_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_dem, gosat_ch4_dem_filtered)

compare_gosat_co2_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_dem, gosat_co2_dem_filtered)
compare_gosat_co2_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_brz, gosat_co2_brz_filtered)

compare_gosat_false_co2_dem, failed = ana.GOSAT_comparison_cakadj_co2(column_compare_co2_dem, gosat_co2_dem_filtered)
compare_gosat_false_co2_brz, failed = ana.GOSAT_comparison_cakadj_co2(column_compare_co2_brz, gosat_co2_brz_filtered)

# %% Check TP perturbation results

# 0.89, 114
sigma_compare_gosat_ch4_tpadj1_brz, n_compare_gosat_ch4_tpadj1_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_tpadj1_brz, 1)
# 1.69, 114
sigma_compare_gosat_ch4_tpadj0_brz, n_compare_gosat_ch4_tpadj0_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_tpadj0_brz, 1)
# 0.98, 113
sigma_compare_gosat_ch4_tpadj1_dem, n_compare_gosat_ch4_tpadj1_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_tpadj1_dem, 1)
# 2.31, 113
sigma_compare_gosat_ch4_tpadj0_dem, n_compare_gosat_ch4_tpadj0_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_tpadj0_dem, 1)

# 0.76, 177
sigma_long_ch4_tpadj1_brz, n_long_ch4_tpadj1_brz = ana.long_error_rms(long_ch4_brz, long_ch4_tpadj1_brz)
# 1.73, 177
sigma_long_ch4_tpadj0_brz, n_long_ch4_tpadj0_brz = ana.long_error_rms(long_ch4_brz, long_ch4_tpadj0_brz)
# 1.16, 186
sigma_long_ch4_tpadj1_dem, n_long_ch4_tpadj1_dem = ana.long_error_rms(long_ch4_dem, long_ch4_tpadj1_dem)
# 2.20, 186
sigma_long_ch4_tpadj0_dem, n_long_ch4_tpadj0_dem = ana.long_error_rms(long_ch4_dem, long_ch4_tpadj0_dem)

# 0.07, 128
sigma_compare_gosat_co2_tpadj1_brz, n_compare_gosat_co2_tpadj1_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_tpadj1_brz, 0)
# 0.08, 128
sigma_compare_gosat_co2_tpadj0_brz, n_compare_gosat_co2_tpadj0_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_tpadj0_brz, 0)
# 0.10, 113
sigma_compare_gosat_co2_tpadj1_dem, n_compare_gosat_co2_tpadj1_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_tpadj1_dem, 0)
# 0.11, 113
sigma_compare_gosat_co2_tpadj0_dem, n_compare_gosat_co2_tpadj0_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_tpadj0_dem, 0)

# 0.06, 176
sigma_long_co2_tpadj1_brz, n_long_co2_tpadj1_brz = ana.long_error_rms(long_co2_brz, long_co2_tpadj1_brz)
# 0.10, 176
sigma_long_co2_tpadj0_brz, n_long_co2_tpadj0_brz = ana.long_error_rms(long_co2_brz, long_co2_tpadj0_brz)
# 0.19, 184
sigma_long_co2_tpadj1_dem, n_long_co2_tpadj1_dem = ana.long_error_rms(long_co2_dem, long_co2_tpadj1_dem)
# 0.22, 184
sigma_long_co2_tpadj0_dem, n_long_co2_tpadj0_dem = ana.long_error_rms(long_co2_dem, long_co2_tpadj0_dem)

# %% Uncertainty 2/5: Tower std (CH4)

column_compare_ch4_tower_adj1_brz, failed_adj_brz = ana.CH4_column_tower_adj(column_tofit_compare_ch4_brz, camsinv_ch4_brz, 1)
column_compare_ch4_tower_adj0_brz, failed_adj_brz = ana.CH4_column_tower_adj(column_tofit_compare_ch4_brz, camsinv_ch4_brz, -1)
column_compare_ch4_tower_adj1_dem, failed_adj_dem = ana.CH4_column_tower_adj(column_tofit_compare_ch4_dem, camsinv_ch4_dem, 1)
column_compare_ch4_tower_adj0_dem, failed_adj_dem = ana.CH4_column_tower_adj(column_tofit_compare_ch4_dem, camsinv_ch4_dem, -1)

compare_gosat_ch4_toweradj1_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_adj1_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_toweradj0_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_adj0_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_toweradj1_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_adj1_dem, gosat_ch4_dem_filtered)
compare_gosat_ch4_toweradj0_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_adj0_dem, gosat_ch4_dem_filtered)

column_long_ch4_tower_adj1_brz, failed_adj_brz = ana.CH4_column_tower_adj(column_tofit_long_ch4_brz, camsinv_ch4_brz, 1)
column_long_ch4_tower_adj0_brz, failed_adj_brz = ana.CH4_column_tower_adj(column_tofit_long_ch4_brz, camsinv_ch4_brz, -1)
column_long_ch4_tower_adj1_dem, failed_adj_dem = ana.CH4_column_tower_adj(column_tofit_long_ch4_dem, camsinv_ch4_dem, 1)
column_long_ch4_tower_adj0_dem, failed_adj_dem = ana.CH4_column_tower_adj(column_tofit_long_ch4_dem, camsinv_ch4_dem, -1)

long_ch4_toweradj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_tower_adj0_brz, 1)
long_ch4_toweradj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_tower_adj1_brz, 1)
long_ch4_toweradj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_tower_adj0_dem, 1)
long_ch4_toweradj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_tower_adj1_dem, 1)

# %% Uncertainty 2/5: Tower adjustment (CO2)

column_compare_co2_tower_adj1_brz, failed_brz = ana.CO2_column_tower_adj(column_tofit_compare_co2_brz, camsinv_co2_brz, 1)
column_compare_co2_tower_adj0_brz, failed_brz = ana.CO2_column_tower_adj(column_tofit_compare_co2_brz, camsinv_co2_brz, -1)
column_compare_co2_tower_adj1_dem, failed_dem = ana.CO2_column_tower_adj(column_tofit_compare_co2_dem, camsinv_co2_dem, 1)
column_compare_co2_tower_adj0_dem, failed_dem = ana.CO2_column_tower_adj(column_tofit_compare_co2_dem, camsinv_co2_dem, -1)

compare_gosat_co2_toweradj1_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_adj1_brz, gosat_co2_brz_filtered)
compare_gosat_co2_toweradj0_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_adj0_brz, gosat_co2_brz_filtered)
compare_gosat_co2_toweradj1_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_adj1_dem, gosat_co2_dem_filtered)
compare_gosat_co2_toweradj0_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_adj0_dem, gosat_co2_dem_filtered)

column_long_co2_tower_adj1_brz, failed_brz = ana.CO2_column_tower_adj(column_tofit_long_co2_brz, camsinv_co2_brz, 1)
column_long_co2_tower_adj0_brz, failed_brz = ana.CO2_column_tower_adj(column_tofit_long_co2_brz, camsinv_co2_brz, -1)
column_long_co2_tower_adj1_dem, failed_dem = ana.CO2_column_tower_adj(column_tofit_long_co2_dem, camsinv_co2_dem, 1)
column_long_co2_tower_adj0_dem, failed_dem = ana.CO2_column_tower_adj(column_tofit_long_co2_dem, camsinv_co2_dem, -1)

long_co2_toweradj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_tower_adj0_brz, 0)
long_co2_toweradj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_tower_adj1_brz, 0)
long_co2_toweradj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_tower_adj0_dem, 0)
long_co2_toweradj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_tower_adj1_dem, 0)

# %% Check tower adjustement results

# 0.35, 36
sigma_compare_gosat_ch4_toweradj1_brz, n_compare_gosat_ch4_toweradj1_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_toweradj1_brz, 1)
# 0.33, 36
sigma_compare_gosat_ch4_toweradj0_brz, n_compare_gosat_ch4_toweradj0_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_toweradj0_brz, 1)
# 0.46, 85
sigma_compare_gosat_ch4_toweradj1_dem, n_compare_gosat_ch4_toweradj1_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_toweradj1_dem, 1)
# 0.49, 85
sigma_compare_gosat_ch4_toweradj0_dem, n_compare_gosat_ch4_toweradj0_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_toweradj0_dem, 1)

# 0.29, 40
sigma_long_ch4_toweradj1_brz, n_long_ch4_toweradj1_brz = ana.long_error_rms(long_ch4_brz, long_ch4_toweradj1_brz)
sigma_long_ch4_toweradj0_brz, n_long_ch4_toweradj0_brz = ana.long_error_rms(long_ch4_brz, long_ch4_toweradj0_brz)
# 0.45, 88
sigma_long_ch4_toweradj1_dem, n_long_ch4_toweradj1_dem = ana.long_error_rms(long_ch4_dem, long_ch4_toweradj1_dem)
sigma_long_ch4_toweradj0_dem, n_long_ch4_toweradj0_dem = ana.long_error_rms(long_ch4_dem, long_ch4_toweradj0_dem) 

# 0.02, 38
sigma_compare_gosat_co2_toweradj1_brz, n_compare_gosat_co2_toweradj1_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_toweradj1_brz, 0)
sigma_compare_gosat_co2_toweradj0_brz, n_compare_gosat_co2_toweradj0_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_toweradj0_brz, 0) 
# 0.02, 85
sigma_compare_gosat_co2_toweradj1_dem, n_compare_gosat_co2_toweradj1_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_toweradj1_dem, 0)
sigma_compare_gosat_co2_toweradj0_dem, n_compare_gosat_co2_toweradj0_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_toweradj0_dem, 0)

# 0.02, 40
sigma_long_co2_toweradj1_brz, n_long_co2_toweradj1_brz = ana.long_error_rms(long_co2_brz, long_co2_toweradj1_brz)
sigma_long_co2_toweradj0_brz, n_long_co2_toweradj0_brz = ana.long_error_rms(long_co2_brz, long_co2_toweradj0_brz) 
# 0.03, 87
sigma_long_co2_toweradj1_dem, n_long_co2_toweradj1_dem = ana.long_error_rms(long_co2_dem, long_co2_toweradj1_dem)
sigma_long_co2_toweradj0_dem, n_long_co2_toweradj0_dem = ana.long_error_rms(long_co2_dem, long_co2_toweradj0_dem)

# %% Uncertainty 3/5: Tower missing

column_compare_ch4_tower_missing_brz, failed_brz = ana.CH4_column_tower_missing(column_tofit_compare_ch4_brz, camsinv_ch4_brz)
column_compare_ch4_tower_missing_dem, failed_dem = ana.CH4_column_tower_missing(column_tofit_compare_ch4_dem, camsinv_ch4_dem)
column_long_ch4_tower_missing_brz, failed_brz = ana.CH4_column_tower_missing(column_tofit_long_ch4_brz, camsinv_ch4_brz)
column_long_ch4_tower_missing_dem, failed_dem = ana.CH4_column_tower_missing(column_tofit_long_ch4_dem, camsinv_ch4_dem)

compare_gosat_ch4_tower_missing_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_missing_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_tower_missing_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_tower_missing_dem, gosat_ch4_dem_filtered)
long_ch4_tower_missing_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_tower_missing_brz, 1)
long_ch4_tower_missing_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_tower_missing_dem, 1)

column_compare_co2_tower_missing_brz, failed_brz = ana.CO2_column_tower_missing(column_tofit_compare_co2_brz, camsinv_co2_brz)
column_compare_co2_tower_missing_dem, failed_dem = ana.CO2_column_tower_missing(column_tofit_compare_co2_dem, camsinv_co2_dem)
column_long_co2_tower_missing_brz, failed_brz = ana.CO2_column_tower_missing(column_tofit_long_co2_brz, camsinv_co2_brz)
column_long_co2_tower_missing_dem, failed_dem = ana.CO2_column_tower_missing(column_tofit_long_co2_dem, camsinv_co2_dem)

compare_gosat_co2_tower_missing_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_missing_brz, gosat_co2_brz_filtered)
compare_gosat_co2_tower_missing_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_tower_missing_dem, gosat_co2_dem_filtered)
long_co2_tower_missing_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_tower_missing_brz, 0)
long_co2_tower_missing_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_tower_missing_dem, 0)

# %% Check tower missing results

# 0.76, 36
sigma_compare_gosat_ch4_towerm_brz, n_compare_gosat_ch4_towerm_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_tower_missing_brz, 1)
# 1.09, 85
sigma_compare_gosat_ch4_towerm_dem, n_compare_gosat_ch4_towerm_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_tower_missing_dem, 1)

# 1.46, 40
sigma_long_ch4_towerm_brz, n_long_ch4_towerm_brz = ana.long_error_rms(long_ch4_brz, long_ch4_tower_missing_brz)
# 1.71, 88
sigma_long_ch4_towerm_dem, n_long_ch4_towerm_dem = ana.long_error_rms(long_ch4_dem, long_ch4_tower_missing_dem)

# 0.04, 38
sigma_compare_gosat_co2_towerm_brz, n_compare_gosat_co2_towerm_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_tower_missing_brz, 0)
# 0.10, 85
sigma_compare_gosat_co2_towerm_dem, n_compare_gosat_co2_towerm_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_tower_missing_dem, 0)

# 0.13, 40
sigma_long_co2_towerm_brz, n_long_co2_towerm_brz = ana.long_error_rms(long_co2_brz, long_co2_tower_missing_brz)
# 0.14, 87
sigma_long_co2_towerm_dem, n_long_co2_towerm_dem = ana.long_error_rms(long_co2_dem, long_co2_tower_missing_dem)

# %% Uncertainty 4/5: Aircraft precision analysis

column_compare_ch4_aj1_brz, failed = ana.CH4_column_aj(column_tofit_compare_ch4_brz, camsinv_ch4_brz, 1)
column_compare_ch4_aj0_brz, failed = ana.CH4_column_aj(column_tofit_compare_ch4_brz, camsinv_ch4_brz, 0)
column_compare_ch4_aj1_dem, failed = ana.CH4_column_aj(column_tofit_compare_ch4_dem, camsinv_ch4_dem, 1)
column_compare_ch4_aj0_dem, failed = ana.CH4_column_aj(column_tofit_compare_ch4_dem, camsinv_ch4_dem, 0)
column_long_ch4_aj1_brz, failed_brz = ana.CH4_column_aj(column_tofit_long_ch4_brz, camsinv_ch4_brz, 1)
column_long_ch4_aj0_brz, failed_brz = ana.CH4_column_aj(column_tofit_long_ch4_brz, camsinv_ch4_brz, 0)
column_long_ch4_aj1_dem, failed_dem = ana.CH4_column_aj(column_tofit_long_ch4_dem, camsinv_ch4_dem, 1)
column_long_ch4_aj0_dem, failed_dem = ana.CH4_column_aj(column_tofit_long_ch4_dem, camsinv_ch4_dem, 0)

compare_gosat_ch4_aj1_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_aj1_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_aj0_brz, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_aj0_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_aj1_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_aj1_dem, gosat_ch4_dem_filtered)
compare_gosat_ch4_aj0_dem, failed = ana.GOSAT_comparison_ch4(column_compare_ch4_aj0_dem, gosat_ch4_dem_filtered)
long_ch4_aj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_aj0_brz, 1)
long_ch4_aj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_ch4_aj1_brz, 1)
long_ch4_aj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_aj0_dem, 1)
long_ch4_aj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_ch4_aj1_dem, 1)

column_compare_co2_aj1_brz, failed = ana.CO2_column_aj(column_tofit_compare_co2_brz, camsinv_co2_brz, 1)
column_compare_co2_aj0_brz, failed = ana.CO2_column_aj(column_tofit_compare_co2_brz, camsinv_co2_brz, 0)
column_compare_co2_aj1_dem, failed = ana.CO2_column_aj(column_tofit_compare_co2_dem, camsinv_co2_dem, 1)
column_compare_co2_aj0_dem, failed = ana.CO2_column_aj(column_tofit_compare_co2_dem, camsinv_co2_dem, 0)
column_long_co2_aj1_brz, failed = ana.CO2_column_aj(column_tofit_long_co2_brz, camsinv_co2_brz, 1)
column_long_co2_aj0_brz, failed = ana.CO2_column_aj(column_tofit_long_co2_brz, camsinv_co2_brz, 0)
column_long_co2_aj1_dem, failed = ana.CO2_column_aj(column_tofit_long_co2_dem, camsinv_co2_dem, 1)
column_long_co2_aj0_dem, failed = ana.CO2_column_aj(column_tofit_long_co2_dem, camsinv_co2_dem, 0)

compare_gosat_co2_aj1_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_aj1_brz, gosat_co2_brz_filtered)
compare_gosat_co2_aj0_brz, failed = ana.GOSAT_comparison_co2(column_compare_co2_aj0_brz, gosat_co2_brz_filtered)
compare_gosat_co2_aj1_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_aj1_dem, gosat_co2_dem_filtered)
compare_gosat_co2_aj0_dem, failed = ana.GOSAT_comparison_co2(column_compare_co2_aj0_dem, gosat_co2_dem_filtered)
long_co2_aj0_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_aj0_brz, 0)
long_co2_aj1_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_long_co2_aj1_brz, 0)
long_co2_aj0_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_aj0_dem, 0)
long_co2_aj1_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_long_co2_aj1_dem, 0)

# %% Check aircraft presicion results

# 1.27, 114
sigma_compare_gosat_ch4_aj1_brz, n_compare_gosat_ch4_aj1_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_aj1_brz, 1)
sigma_compare_gosat_ch4_aj0_brz, n_compare_gosat_ch4_aj0_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_aj0_brz, 1)
# 1.26, 113
sigma_compare_gosat_ch4_aj1_dem, n_compare_gosat_ch4_aj1_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_aj1_dem, 1)
sigma_compare_gosat_ch4_aj0_dem, n_compare_gosat_ch4_aj0_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_aj0_dem, 1)

# 1.24, 177
sigma_long_ch4_aj1_brz, n_long_ch4_aj1_brz = ana.long_error_rms(long_ch4_brz, long_ch4_aj1_brz)
sigma_long_ch4_aj0_brz, n_long_ch4_aj0_brz = ana.long_error_rms(long_ch4_brz, long_ch4_aj0_brz)
# 1.21, 186
sigma_long_ch4_aj1_dem, n_long_ch4_aj1_dem = ana.long_error_rms(long_ch4_dem, long_ch4_aj1_dem)
sigma_long_ch4_aj0_dem, n_long_ch4_aj0_dem = ana.long_error_rms(long_ch4_dem, long_ch4_aj0_dem) 

# 0.22, 128
sigma_compare_gosat_co2_aj1_brz, n_compare_gosat_co2_aj1_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_aj1_brz, 0)
sigma_compare_gosat_co2_aj0_brz, n_compare_gosat_co2_aj0_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_aj0_brz, 0) 
# 0.22, 113
sigma_compare_gosat_co2_aj1_dem, n_compare_gosat_co2_toweradj1_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_aj1_dem, 0)
sigma_compare_gosat_co2_aj0_dem, n_compare_gosat_co2_toweradj0_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_aj0_dem, 0)

# 0.22, 176
sigma_long_co2_aj1_brz, n_long_co2_aj1_brz = ana.long_error_rms(long_co2_brz, long_co2_aj1_brz)
sigma_long_co2_aj0_brz, n_long_co2_aj0_brz = ana.long_error_rms(long_co2_brz, long_co2_aj0_brz) 
# 0.21, 184
sigma_long_co2_aj1_dem, n_long_co2_aj1_dem = ana.long_error_rms(long_co2_dem, long_co2_aj1_dem)
sigma_long_co2_aj0_dem, n_long_co2_aj0_dem = ana.long_error_rms(long_co2_dem, long_co2_aj0_dem)


# %% Uncertainty 5/5: Stratosphere replacement

column_str_adj_ch4_brz, failed_brz = ana.CH4_strat_adjust(column_tofit_compare_ch4_brz, scisat_brz, camsinv_ch4_brz)
column_str_adj_ch4_dem, failed_dem = ana.CH4_strat_adjust(column_tofit_compare_ch4_dem, scisat_dem, camsinv_ch4_dem)
column_str_adj_co2_brz, failed_brz = ana.CO2_strat_adjust(column_tofit_compare_co2_brz, scisat_brz, camsinv_co2_brz)
column_str_adj_co2_dem, failed_dem = ana.CO2_strat_adjust(column_tofit_compare_co2_dem, scisat_dem, camsinv_co2_dem)

compare_gosat_ch4_str_adj_brz, failed = ana.GOSAT_comparison_str_adj_ch4(column_str_adj_ch4_brz, gosat_ch4_brz_filtered)
compare_gosat_ch4_str_adj_dem, failed = ana.GOSAT_comparison_str_adj_ch4(column_str_adj_ch4_dem, gosat_ch4_dem_filtered)
compare_gosat_co2_str_adj_brz, failed = ana.GOSAT_comparison_co2(column_str_adj_co2_brz, gosat_co2_brz_filtered)
compare_gosat_co2_str_adj_dem, failed = ana.GOSAT_comparison_co2(column_str_adj_co2_dem, gosat_co2_dem_filtered)

column_str_adj_ch4_brz, failed_brz = ana.CH4_strat_adjust(column_tofit_long_ch4_brz, scisat_brz, camsinv_ch4_brz)
column_str_adj_ch4_dem, failed_dem = ana.CH4_strat_adjust(column_tofit_long_ch4_dem, scisat_dem, camsinv_ch4_dem)
column_str_adj_co2_brz, failed_brz = ana.CO2_strat_adjust(column_tofit_long_co2_brz, scisat_brz, camsinv_co2_brz)
column_str_adj_co2_dem, failed_dem = ana.CO2_strat_adjust(column_tofit_long_co2_dem, scisat_dem, camsinv_co2_dem)

long_ch4_str_adj_brz, failed = ana.cal_merra2_str_adj_ghg(merra2_dry_air_mass_brz, column_str_adj_ch4_brz, 1)
long_ch4_str_adj_dem, failed = ana.cal_merra2_str_adj_ghg(merra2_dry_air_mass_dem, column_str_adj_ch4_dem, 1)
long_co2_str_adj_brz, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_brz, column_str_adj_co2_brz, 0)
long_co2_str_adj_dem, failed = ana.cal_merra2_ghg(merra2_dry_air_mass_dem, column_str_adj_co2_dem, 0)

# %% Check stratosphere replacement results

# 6.61, 78
sigma_compare_gosat_ch4_str_adj_brz, n_compare_gosat_ch4_str_adj_brz = ana.compare_gosat_error_rms(compare_gosat_ch4_brz, compare_gosat_ch4_str_adj_brz, 1)
# 5.72, 80
sigma_compare_gosat_ch4_str_adj_dem, n_compare_gosat_ch4_str_adj_dem = ana.compare_gosat_error_rms(compare_gosat_ch4_dem, compare_gosat_ch4_str_adj_dem, 1)

# 9.31, 75
sigma_long_ch4_str_adj_brz, n_long_ch4_str_adj_brz = ana.long_error_rms(long_ch4_brz, long_ch4_str_adj_brz)
# 11.25, 92
sigma_long_ch4_str_adj_dem, n_long_ch4_str_adj_dem = ana.long_error_rms(long_ch4_dem, long_ch4_str_adj_dem)

# 1.73, 86
sigma_compare_gosat_co2_str_adj_brz, n_compare_gosat_co2_str_adj_brz = ana.compare_gosat_error_rms(compare_gosat_co2_brz, compare_gosat_co2_str_adj_brz, 0)
# 1.44, 80
sigma_compare_gosat_co2_str_adj_dem, n_compare_gosat_co2_str_adj_dem = ana.compare_gosat_error_rms(compare_gosat_co2_dem, compare_gosat_co2_str_adj_dem, 0)

# 2.85, 75
sigma_long_co2_str_adj_brz, n_long_co2_str_adj_brz = ana.long_error_rms(long_co2_brz, long_co2_str_adj_brz)
# 2.75, 91
sigma_long_co2_str_adj_dem, n_long_co2_str_adj_dem = ana.long_error_rms(long_co2_dem, long_co2_str_adj_dem)

# %% Uncertainty analysis summary

## CO2
 # 1.76
sigma_compare_gosat_unctower_co2_brz = np.sqrt(sigma_compare_gosat_co2_toweradj0_brz **2 + sigma_compare_gosat_co2_toweradj1_brz **2
                                            + sigma_compare_gosat_co2_str_adj_brz **2 
                                            + sigma_compare_gosat_co2_tpadj0_brz **2 + sigma_compare_gosat_co2_tpadj1_brz **2
                                            + sigma_compare_gosat_co2_aj1_brz **2 + sigma_compare_gosat_co2_aj0_brz **2)
 # 1.76
sigma_compare_gosat_notower_co2_brz = np.sqrt(sigma_compare_gosat_co2_towerm_brz **2 + sigma_compare_gosat_co2_str_adj_brz **2
                                              + sigma_compare_gosat_co2_tpadj0_brz **2 + sigma_compare_gosat_co2_tpadj1_brz **2
                                              + sigma_compare_gosat_co2_aj0_brz **2 + sigma_compare_gosat_co2_aj1_brz **2)
 # 1.48
sigma_compare_gosat_unctower_co2_dem = np.sqrt(sigma_compare_gosat_co2_toweradj0_dem **2 + sigma_compare_gosat_co2_toweradj1_dem **2
                                            + sigma_compare_gosat_co2_str_adj_dem **2
                                            + sigma_compare_gosat_co2_tpadj0_dem **2 + sigma_compare_gosat_co2_tpadj1_dem **2
                                            + sigma_compare_gosat_co2_aj0_dem **2 + sigma_compare_gosat_co2_aj1_dem **2)
 # 1.48
sigma_compare_gosat_notower_co2_dem = np.sqrt(sigma_compare_gosat_co2_towerm_dem**2 + sigma_compare_gosat_co2_str_adj_dem **2
                                            + sigma_compare_gosat_co2_tpadj0_dem **2 + sigma_compare_gosat_co2_tpadj1_dem **2
                                            + sigma_compare_gosat_co2_aj0_dem **2 + sigma_compare_gosat_co2_aj1_dem **2)
 # 2.86
sigma_long_unctower_co2_brz = np.sqrt(sigma_long_co2_toweradj0_brz **2 + sigma_long_co2_toweradj1_brz **2
                                      + sigma_long_co2_aj0_brz **2 + sigma_long_co2_aj1_brz **2
                                      + sigma_long_co2_str_adj_brz **2
                                      + sigma_long_co2_tpadj0_brz **2 + sigma_long_co2_tpadj1_brz **2)
 # 2.87
sigma_long_notower_co2_brz = np.sqrt(sigma_long_co2_towerm_brz **2 + sigma_long_co2_str_adj_brz **2
                                     + sigma_long_co2_aj0_brz **2 + sigma_long_co2_aj1_brz **2
                                     + sigma_long_co2_tpadj0_brz **2 + sigma_long_co2_tpadj1_brz **2)
 # 2.78
sigma_long_unctower_co2_dem = np.sqrt(sigma_long_co2_toweradj0_dem **2 + sigma_long_co2_toweradj1_dem **2
                                      + sigma_long_co2_str_adj_dem **2
                                      + sigma_long_co2_aj0_dem **2 + sigma_long_co2_aj1_dem **2
                                      + sigma_long_co2_tpadj0_dem **2 + sigma_long_co2_tpadj1_dem **2)
 # 2.78
sigma_long_notower_co2_dem = np.sqrt(sigma_long_co2_towerm_dem**2 + sigma_long_co2_str_adj_dem **2
                                     + sigma_long_co2_aj0_dem**2 + sigma_long_co2_aj1_dem**2
                                     + sigma_long_co2_tpadj0_dem **2 + sigma_long_co2_tpadj1_dem **2)

## CH4
 # 7.13
sigma_compare_gosat_unctower_ch4_brz = np.sqrt(sigma_compare_gosat_ch4_toweradj0_brz **2 + sigma_compare_gosat_ch4_toweradj1_brz **2
                                            + sigma_compare_gosat_ch4_aj0_brz **2 + sigma_compare_gosat_ch4_aj1_brz **2
                                            + sigma_compare_gosat_ch4_str_adj_brz **2 + sigma_compare_gosat_ch4_tpadj0_brz **2 
                                            + sigma_compare_gosat_ch4_tpadj1_brz **2)
 # 7.15
sigma_compare_gosat_notower_ch4_brz = np.sqrt(sigma_compare_gosat_ch4_towerm_brz **2
                                            + sigma_compare_gosat_ch4_aj0_brz **2 + sigma_compare_gosat_ch4_aj1_brz **2
                                            + sigma_compare_gosat_ch4_str_adj_brz **2 + sigma_compare_gosat_ch4_tpadj0_brz **2 
                                            + sigma_compare_gosat_ch4_tpadj1_brz **2)
 # 6.53
sigma_compare_gosat_unctower_ch4_dem = np.sqrt(sigma_compare_gosat_ch4_toweradj0_dem **2 + sigma_compare_gosat_ch4_toweradj1_dem **2
                                            + sigma_compare_gosat_ch4_aj0_dem **2 + sigma_compare_gosat_ch4_aj1_dem **2
                                            + sigma_compare_gosat_ch4_str_adj_dem **2 + sigma_compare_gosat_ch4_tpadj0_dem **2 
                                            + sigma_compare_gosat_ch4_tpadj1_dem **2)
 # 6.59
sigma_compare_gosat_notower_ch4_dem = np.sqrt(sigma_compare_gosat_ch4_towerm_dem**2
                                            + sigma_compare_gosat_ch4_aj0_dem **2 + sigma_compare_gosat_ch4_aj1_dem **2
                                            + sigma_compare_gosat_ch4_str_adj_dem **2 + sigma_compare_gosat_ch4_tpadj0_dem **2 
                                            + sigma_compare_gosat_ch4_tpadj1_dem **2)
 # 9.67
sigma_long_unctower_ch4_brz = np.sqrt(sigma_long_ch4_toweradj0_brz **2 + sigma_long_ch4_toweradj1_brz **2
                                      + sigma_long_ch4_aj0_brz **2 + sigma_long_ch4_aj1_brz **2
                                      + sigma_long_ch4_str_adj_brz **2 + sigma_long_ch4_tpadj0_brz **2 
                                      + sigma_long_ch4_tpadj1_brz **2)
 # 9.77
sigma_long_notower_ch4_brz = np.sqrt(sigma_long_ch4_towerm_brz **2
                                     + sigma_long_ch4_aj0_brz **2 + sigma_long_ch4_aj1_brz **2
                                     + sigma_long_ch4_str_adj_brz **2 + sigma_long_ch4_tpadj0_brz **2 
                                     + sigma_long_ch4_tpadj1_brz **2)
 # 11.67
sigma_long_unctower_ch4_dem = np.sqrt(sigma_long_ch4_toweradj0_dem **2 + sigma_long_ch4_toweradj1_dem **2
                                      + sigma_long_ch4_aj0_dem **2 + sigma_long_ch4_aj1_dem **2
                                      + sigma_long_ch4_str_adj_dem **2 + sigma_long_ch4_tpadj0_dem **2 
                                      + sigma_long_ch4_tpadj1_dem **2)
 # 11.78
sigma_long_notower_ch4_dem = np.sqrt(sigma_long_ch4_towerm_dem **2 +
                                     + sigma_long_ch4_aj0_dem **2 + sigma_long_ch4_aj1_dem **2
                                     + sigma_long_ch4_str_adj_dem **2 + sigma_long_ch4_tpadj0_dem **2 
                                     + sigma_long_ch4_tpadj1_dem **2)

# %% Load TCCON data

# Ny-Ålesund, GMT+1
# tccon_ny = xr.open_dataset(tccon_dir + 'ny20050316_20250417.public.qc.nc')
# Eureka, GMT-5
# tccon_eu = xr.open_dataset(tccon_dir + 'eu20100725_20200707.public.qc.nc')
# Sodankylä, GMT+3
tccon_so = xr.open_dataset(tccon_dir + 'so20090516_20230530.public.qc.nc')
# 
# Bialystok, GMT+1
tccon_bi = xr.open_dataset(tccon_dir + 'bi20090313_20181001.public.qc.nc')
# Bremen, GMT+1
tccon_br = xr.open_dataset(tccon_dir + 'br20060213_20250402.public.qc.nc')
# East Trout Lake, GMT-5
tccon_es = xr.open_dataset(tccon_dir + 'et20161003_20260510.public.qc.nc')
# Park Falls, GMT-6
tccon_pa = xr.open_dataset(tccon_dir + 'pa20040602_20251223.public.qc.nc')

del tccon_dir

# %% Detrend TCCON data

tccon_fit_so, tccon_res_so, tccon_amp_co2_so, tccon_amp_ch4_so = ana.thoning_noaa(tccon_so, 10)
tccon_fit_bi, tccon_res_bi, tccon_amp_co2_bi, tccon_amp_ch4_bi = ana.thoning_noaa(tccon_bi, 12)
tccon_fit_br, tccon_res_br, tccon_amp_co2_br, tccon_amp_ch4_br = ana.thoning_noaa(tccon_br, 12)
tccon_fit_pa, tccon_res_pa, tccon_amp_co2_pa, tccon_amp_ch4_pa = ana.thoning_noaa(tccon_pa, 19)
tccon_fit_es, tccon_res_es, tccon_amp_co2_es, tccon_amp_ch4_es = ana.thoning_noaa(tccon_es, 18)

# Annual growth rate
tccon_grw_so = tccon_fit_so.groupby('Year').agg({'trend_co2': 'first', 'trend_ch4': 'first'})
# np.diff(tccon_grw_so.trend_co2).mean()
tccon_grw_bi = tccon_fit_bi.groupby('Year').agg({'trend_co2': 'first', 'trend_ch4': 'first'})
tccon_grw_br = tccon_fit_br.groupby('Year').agg({'trend_co2': 'first', 'trend_ch4': 'first'})
tccon_grw_pa = tccon_fit_pa.groupby('Year').agg({'trend_co2': 'first', 'trend_ch4': 'first'})
tccon_grw_es = tccon_fit_es.groupby('Year').agg({'trend_co2': 'first', 'trend_ch4': 'first'})

# CH4 summer growth rate
tccon_grws_so = tccon_fit_so[(tccon_fit_so.Season==2) & (tccon_fit_so.Year<=2019) & (tccon_fit_so.Year>=2009)].groupby('Year').agg({'trend_ch4':'mean'})
tccon_grws_bi = tccon_fit_bi[(tccon_fit_bi.Season==2) & (tccon_fit_bi.Year<=2019) & (tccon_fit_bi.Year>=2009)].groupby('Year').agg({'trend_ch4':'mean'})
tccon_grws_br = tccon_fit_br[(tccon_fit_br.Season==2) & (tccon_fit_br.Year<=2019) & (tccon_fit_br.Year>=2009)].groupby('Year').agg({'trend_ch4':'mean'})

tccon_grws_so = tccon_fit_so[(tccon_fit_so.Season==2) & (tccon_fit_so.Year>=2011)].groupby('Year').agg({'trend_ch4':'mean'})
tccon_grws_bi = tccon_fit_bi[(tccon_fit_bi.Season==2) & (tccon_fit_bi.Year>=2011)].groupby('Year').agg({'trend_ch4':'mean'})
tccon_grws_br = tccon_fit_br[(tccon_fit_br.Season==2) & (tccon_fit_br.Year>=2011)].groupby('Year').agg({'trend_ch4':'mean'})


# %% Detrend BRZ and DEM

long_co2_fit_brz, long_co2_res_brz, long_co2_amp_brz = ana.thoning_noaa_local(long_co2_brz)
long_co2_fit_dem, long_co2_res_dem, long_co2_amp_dem = ana.thoning_noaa_local(long_co2_dem)
long_ch4_fit_brz, long_ch4_res_brz, long_ch4_amp_brz = ana.thoning_noaa_local(long_ch4_brz)
long_ch4_fit_dem, long_ch4_res_dem, long_ch4_amp_dem = ana.thoning_noaa_local(long_ch4_dem)

# Annual growth rate
long_co2_grw_dem = long_co2_fit_dem.groupby('Year').agg({'trend':'mean'}).dropna()
long_ch4_grw_dem = long_ch4_fit_dem.groupby('Year').agg({'trend':'mean'}).dropna()
long_co2_grw_brz = long_co2_fit_brz.groupby('Year').agg({'trend':'mean'}).dropna()
long_ch4_grw_brz = long_ch4_fit_brz.groupby('Year').agg({'trend':'mean'}).dropna()

# CH4 summer growth rate
long_ch4_grws_dem = long_ch4_fit_dem[long_ch4_fit_dem.Season==2].groupby('Year').agg({'trend':'mean'}).dropna()
np.diff(long_ch4_grws_dem.trend[9:]).mean()
long_ch4_grws_brz = long_ch4_fit_brz[long_ch4_fit_brz.Season==2].groupby('Year').agg({'trend':'mean'}).dropna()
np.diff(long_ch4_grws_brz.trend[9:]).mean()

# %% Anomaly and error detection

# dem anomalies and errors
compare_gosat_ch4_dem = ana.add_sigma_to_fill(compare_gosat_ch4_dem, sigma_compare_gosat_unctower_ch4_dem, sigma_compare_gosat_notower_ch4_dem)
out_ch4_dem, norm_ch4_dem, compare_bounds = ana.detect_outliers_iqr(compare_gosat_ch4_dem, 'Diff', 'sigma')
long_ch4_dem = ana.add_sigma_to_fill(long_ch4_res_dem, sigma_long_unctower_ch4_dem, sigma_long_notower_ch4_dem)
anom_ch4_dem, reg_ch4_dem, long_bounds = ana.detect_outliers_iqr(long_ch4_dem, 'detrend', 'sigma')

compare_gosat_co2_dem = ana.add_sigma_to_fill(compare_gosat_co2_dem, sigma_compare_gosat_unctower_co2_dem, sigma_compare_gosat_notower_co2_dem)
out_co2_dem, norm_co2_dem, compare_bounds = ana.detect_outliers_iqr(compare_gosat_co2_dem, 'Diff', 'sigma')
long_co2_dem = ana.add_sigma_to_fill(long_co2_res_dem, sigma_long_unctower_co2_dem, sigma_long_notower_co2_dem)
anom_co2_dem, reg_co2_dem, long_bounds = ana.detect_outliers_iqr(long_co2_dem, 'detrend', 'sigma')

# brz anomalies and errors
compare_gosat_ch4_brz = ana.add_sigma_to_fill(compare_gosat_ch4_brz, sigma_compare_gosat_unctower_ch4_brz, sigma_compare_gosat_notower_ch4_brz)
out_ch4_brz, norm_ch4_brz, compare_bounds = ana.detect_outliers_iqr(compare_gosat_ch4_brz, 'Diff', 'sigma')
long_ch4_brz = ana.add_sigma_to_fill(long_ch4_res_brz, sigma_long_unctower_ch4_brz, sigma_long_notower_ch4_brz)
anom_ch4_brz, reg_ch4_brz, long_bounds = ana.detect_outliers_iqr(long_ch4_brz, 'detrend', 'sigma')

compare_gosat_co2_brz = ana.add_sigma_to_fill(compare_gosat_co2_brz, sigma_compare_gosat_unctower_co2_brz, sigma_compare_gosat_notower_co2_brz)
out_co2_brz, norm_co2_brz, compare_bounds = ana.detect_outliers_iqr(compare_gosat_co2_brz, 'Diff', 'sigma')
long_co2_brz = ana.add_sigma_to_fill(long_co2_res_brz, sigma_long_unctower_co2_brz, sigma_long_notower_co2_brz)
anom_co2_brz, reg_co2_brz, long_bounds = ana.detect_outliers_iqr(long_co2_brz, 'detrend', 'sigma')

# %% Remove all column adjusting dataframes to save storage

del long_co2_brz, long_ch4_brz, long_co2_dem, long_ch4_dem, long_bounds, P0

# Run the following in the console without #
# %reset_selective -f compare_gosat_co2_t
# %reset_selective -f compare_gosat_ch4_t
# %reset_selective -f compare_gosat_co2_a
# %reset_selective -f compare_gosat_ch4_a
# %reset_selective -f compare_gosat_co2_s
# %reset_selective -f compare_gosat_ch4_s
# %reset_selective -f column_long_co2_t
# %reset_selective -f column_long_ch4_t
# %reset_selective -f column_long_co2_a
# %reset_selective -f column_long_ch4_a
# %reset_selective -f long_co2_a
# %reset_selective -f long_ch4_a
# %reset_selective -f long_co2_t
# %reset_selective -f long_ch4_t
# %reset_selective -f long_co2_s
# %reset_selective -f long_ch4_s
# %reset_selective -f column_str
# %reset_selective -f fail

# %% GOSAT compare with TCCON

# Generate CO2 Comparisons
# compare_co2_eu = ana.compare_tg(gosat_co2_eu, tccon_res_eu, 'co2')
compare_co2_so = ana.compare_tg(gosat_co2_so, tccon_res_so, 'co2')
compare_co2_es = ana.compare_tg(gosat_co2_es, tccon_res_es, 'co2')
compare_co2_bi = ana.compare_tg(gosat_co2_bi, tccon_res_bi, 'co2')
compare_co2_br = ana.compare_tg(gosat_co2_br, tccon_res_br, 'co2')
compare_co2_pa = ana.compare_tg(gosat_co2_pa, tccon_res_pa, 'co2')

# Generate CH4 Comparisons
# compare_ch4_eu = ana.compare_tg(gosat_ch4_eu, tccon_res_eu, 'ch4')
compare_ch4_so = ana.compare_tg(gosat_ch4_so, tccon_res_so, 'ch4')
compare_ch4_es = ana.compare_tg(gosat_ch4_es, tccon_res_es, 'ch4')
compare_ch4_bi = ana.compare_tg(gosat_ch4_bi, tccon_res_bi, 'ch4')
compare_ch4_br = ana.compare_tg(gosat_ch4_br, tccon_res_br, 'ch4')
compare_ch4_pa = ana.compare_tg(gosat_ch4_pa, tccon_res_pa, 'ch4')

# %% 
so = abs(compare_ch4_so[-1])
et = abs(compare_ch4_es[-1])
br = abs(compare_ch4_br[-1])
bi = abs(compare_ch4_bi[-1])
pa = abs(compare_ch4_pa[-1])
tccons = np.concatenate((so,et,br,bi,pa))
np.mean(tccons)
np.std(tccons)

dem = norm_ch4_dem.Diff.abs()/norm_ch4_dem.CH4_gosat
brz = norm_ch4_brz.Diff.abs()/norm_ch4_brz.CH4_gosat
wss = np.concatenate((dem,brz))
np.mean(wss)
np.std(wss)

t_stat, p_val = stats.ttest_ind(tccons, wss, equal_var=False)
p_val

n_A, n_B = len(tccons), len(wss)
v_a = (tccons.std() ** 2) / len(tccons)
v_b = (wss.std() ** 2) / len(wss)
df = (v_a + v_b) ** 2 / ((v_a ** 2) / (n_A - 1) + (v_b ** 2) / (n_B - 1))

s_pooled = np.sqrt(
    ((n_A - 1) * tccons.std()**2 + (n_B - 1) * wss.std()**2) / (n_A + n_B - 2)
)
cohens_d = (wss.mean() - tccons.mean()) / s_pooled

# Exact Hedges' correction factor
j_correction = 1 - (3 / (4 * (n_A + n_B) - 9))
hedges_g = cohens_d * j_correction

print(f"\nEffect Size:")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Hedges' g: {hedges_g:.4f}")