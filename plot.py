#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures
"""
# %% Load packages and define plotting parameters

import pandas as pd
import numpy as np
from datetime import datetime
import ast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

import analysis as ana

ic_cm = 1/2.54
plt.rcParams.update({
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "xtick.bottom": True,
    "ytick.left": True, "ytick.right": False,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5
})

# %% Define saving directory

save_dir = "/Users/qnqin/Library/CloudStorage/Box-Box/Qin/Fig_Qin/"


# %% Figure A1. Barometric curve fitting

df1 = igra_param_dem
df2 = igra_param_nov

fig, axs = plt.subplots(2,2, figsize=(18*ic_cm,15*ic_cm*.8), dpi=300, sharey='row')

alts = np.arange(.01, 35, .5)

# Plot curves and original data points
for i in range(len(df1)):
    test = df1.iloc[i]
    prs = test.P0 * np.exp((-test.a * alts / test.H) + test.b)
    axs[0,0].plot(alts, prs, color='grey', linewidth=.3, alpha=.3)
    axs[0,0].scatter(test.Altitude, test.Pressure, color='coral', s=.5, alpha=.3, zorder=3)

for i in range(len(df2)):
    test = df2.iloc[i]
    prs = test.P0 * np.exp((-test.a * alts / test.H) + test.b)
    axs[0,1].plot(alts, prs, color='grey', linewidth=.3, alpha=.5)
    axs[0,1].scatter(test.Altitude, test.Pressure, color='coral', s=.5, alpha=.3, zorder=3)

# Plot RMSE
axs[1,0].hist(df1.rmse, bins=range(int(min(df1.rmse)), int(max(df1.rmse)) + 1, 1), color='skyblue')
axs[1,1].hist(df2.rmse, bins=range(int(min(df2.rmse)), int(max(df2.rmse)) + 1, 1) , color='skyblue')

axs[0,0].set_xlabel('Altitude [km]')
axs[0,1].set_xlabel('Altitude [km]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Frequency')
axs[1,0].set_xlabel('RMSE [hPa]')
axs[1,1].set_xlabel('RMSE [hPa]')

axs[0,0].set_ylim(1050,-10)
legend_elements = [
    Line2D([0], [0], color='grey', linewidth=.3, alpha=.3, label='Fitted curve'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='coral',
           markersize=5, alpha=.3, label='Original data')
]
axs[0,1].legend(handles=legend_elements, ncol=1, loc='lower right', fontsize=6, handletextpad=0.05,
                columnspacing=0.1, markerscale=.5, borderaxespad=1)
axs[0,0].text(0, 90, "(a) DEM", fontsize=10)
axs[0,1].text(0, 90, "(b) BRZ", fontsize=10)
axs[1,0].text(0, 2620, "(c) DEM", fontsize=10)
axs[1,1].text(0, 2620, "(d) BRZ", fontsize=10)

plt.tight_layout()
plt.savefig(save_dir + 'Fig_A1.png', bbox_inches='tight', transparent=True)

# %% Figure B1. MERRA-2 TP vs. IGRA TP
fig, axs = plt.subplots(1,2, figsize=(18*ic_cm,15*ic_cm*.5), dpi=300, sharey='row', sharex='row')

df2 = tp_cal_nov
df1 = tp_cal_sur
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']
seasons = ['MAM','JJA','SON','DJF']
prs = np.arange(0, max(df1.TP_merra2), .5)

for s in [1,2,3,4]:
    test = df1[df1.Season == s]
    axs[0].scatter(test.TP_merra2, test.TP_igra, c=colors[s-1], alpha=.3, s=.5, label=seasons[s-1])
    test = df2[df2.Season == s]
    axs[1].scatter(test.TP_merra2, test.TP_igra, c=colors[s-1], alpha=.3, s=.5, label=seasons[s-1])

df2 = tp_reg_nov
df1 = tp_reg_sur

for s in [1,2,3,4]:
    test = df1.iloc[s-1]
    axs[0].plot(prs, test.slope * prs + test.int, c=colors[s-1], alpha=.8, linewidth=.5, label=seasons[s-1])
    test = df2.iloc[s-1]
    axs[1].plot(prs, test.slope * prs + test.int, c=colors[s-1], alpha=.8, linewidth=.5, label=seasons[s-1])

axs[0].axline(xy1 = (400, 400), slope = 1,
              color='black', linestyle='--', linewidth=.5, label='1:1 line')
axs[1].axline(xy1 = (400, 400), slope = 1,
              color='black', linestyle='--', linewidth=.5, label='1:1 line')

# Formatting
axs[0].set_xlabel("MERRA-2 [hPa]")
axs[1].set_xlabel("MERRA-2 [hPa]")
axs[0].set_ylabel("IGRA [hPa]")
axs[0].text(0, 525, "(a) DEM", fontsize=10)
axs[1].text(0, 525, "(b) BRZ", fontsize=10)
axs[1].legend(ncol=1, loc='lower right', fontsize=6, handletextpad=0.05,
              columnspacing=0.1, markerscale=.5, borderaxespad=1)

plt.tight_layout()
plt.savefig(save_dir + 'Fig_B1.png', bbox_inches='tight', transparent=True)

# %% Figure 3. Profile examples

fig, axs = plt.subplots(2,2,figsize=(18*ic_cm*.7,15*ic_cm), sharey='row', sharex='row', dpi=300)

df1 = column_long_co2_brz
df2 = column_long_ch4_dem

# CO2 BRZ summer
this_test = df1.iloc[115]
p = np.append(this_test.Pressure_1, this_test.Pressure_2)
ghg = np.append(this_test.CO2_1, this_test.CO2_2)
sort_indices = np.argsort(p)
sorted_p = p[sort_indices]
sorted_ghg = ghg[sort_indices]
axs[0,0].plot(sorted_ghg, sorted_p, c='royalblue', lw=1.2, zorder=2)

axs[0,0].scatter(this_test.CO2_tower, this_test.Pressure_tower, c='grey',
               marker='x', label='Tower', s=10, zorder=3)
axs[0,0].scatter(this_test.CO2_aircraft, this_test.Pressure_aircraft, c='grey',
               label='Aircraft', s=8, zorder=3)
axs[0,0].scatter(this_test.CO2_camsinv[-40:], this_test.Pressure_camsinv[-40:],
               c='grey', marker='d', label='CAMSinv', s=6, zorder=3)

axs[0,0].axhline(y=this_test.Pressure_TP, color='black', linestyle='dashdot', linewidth=.5, label='Tropopause')
axs[0,0].axhline(y=this_test.Pressure_PBL, color='black', linestyle='--', linewidth=.5, label='PBL')
axs[0,0].text(412, 40, "(a)", fontsize=10)

# CO2 BRZ winter
this_test = df1.iloc[110]
p = np.append(this_test.Pressure_1, this_test.Pressure_2)
ghg = np.append(this_test.CO2_1, this_test.CO2_2)
sort_indices = np.argsort(p)
sorted_p = p[sort_indices]
sorted_ghg = ghg[sort_indices]
axs[0,1].plot(sorted_ghg, sorted_p, c='royalblue', lw=1.2, zorder=2)

axs[0,1].scatter(this_test.CO2_tower, this_test.Pressure_tower, c='grey',
               marker='x', label='Tower', s=10, zorder=3)
axs[0,1].scatter(this_test.CO2_aircraft, this_test.Pressure_aircraft, c='grey',
               label='Aircraft', s=8, zorder=3)
axs[0,1].scatter(this_test.CO2_camsinv[-42:], this_test.Pressure_camsinv[-42:],
               c='grey', marker='d', label='CAMSinv', s=6, zorder=3)

axs[0,1].axhline(y=this_test.Pressure_TP, color='black', linestyle='dashdot', linewidth=.5, label='Tropopause')
axs[0,1].axhline(y=this_test.Pressure_PBL, color='black', linestyle='--', linewidth=.5, label='PBL')
axs[0,1].text(412, 40, "(b)", fontsize=10)

# CH4 DEM summer
this_test = df2.iloc[132]
p = np.append(np.append(this_test.Pressure_1, this_test.Pressure_2), this_test.Pressure_0)
ghg = np.append(np.append(this_test.CH4_1, this_test.CH4_2), this_test.CH4_0)
sort_indices = np.argsort(p)
sorted_p = p[sort_indices]
sorted_ghg = ghg[sort_indices]
axs[1,0].plot(sorted_ghg, sorted_p, c='mediumpurple', lw=1.2, zorder=2)

axs[1,0].scatter(this_test.CH4_tower, this_test.Pressure_tower, c='grey',
               marker='x', label='Tower', s=10, zorder=3)
axs[1,0].scatter(this_test.CH4_aircraft, this_test.Pressure_aircraft, c='grey',
               label='Aircraft', s=8, zorder=3)
axs[1,0].scatter(this_test.CH4_camsinv[-16:], this_test.Pressure_camsinv[-16:],
               c='grey', marker='d', label='CAMSinv', s=6, zorder=3)

axs[1,0].axhline(y=this_test.Pressure_TP, color='black', linestyle='dashdot', linewidth=.5, label='Tropopause')
axs[1,0].axhline(y=this_test.Pressure_PBL, color='black', linestyle='--', linewidth=.5, label='PBL')
axs[1,0].text(1966,40, "(c)", fontsize=10)

# CH4 DEM winter
this_test = df2.iloc[127]
p = np.append(np.append(this_test.Pressure_1, this_test.Pressure_2), this_test.Pressure_0)
ghg = np.append(np.append(this_test.CH4_1, this_test.CH4_2), this_test.CH4_0)
sort_indices = np.argsort(p)
sorted_p = p[sort_indices]
sorted_ghg = ghg[sort_indices]
axs[1,1].plot(sorted_ghg, sorted_p, c='mediumpurple', lw=1.2, zorder=2)

axs[1,1].scatter(this_test.CH4_tower, this_test.Pressure_tower, c='grey',
               marker='x', label='Tower', s=10, zorder=3)
axs[1,1].scatter(this_test.CH4_aircraft, this_test.Pressure_aircraft, c='grey',
               label='Aircraft', s=8, zorder=3)
axs[1,1].scatter(this_test.CH4_camsinv[-16:], this_test.Pressure_camsinv[-16:],
               c='grey', marker='d', label='CAMSinv', s=6, zorder=3)

axs[1,1].axhline(y=this_test.Pressure_TP, color='black', linestyle='dashdot', linewidth=.5, label='Tropopause')
axs[1,1].axhline(y=this_test.Pressure_PBL, color='black', linestyle='--', linewidth=.5, label='PBL')
axs[1,1].text(1966,40, "(d)", fontsize=10)


axs[1,0].legend(ncol=1, fontsize=6, columnspacing=0.1, borderaxespad=1, loc=3)
axs[0,0].set_xlabel('CO$_2$ [ppm]')
axs[0,1].set_xlabel('CO$_2$ [ppm]')
axs[1,0].set_xlabel('CH$_4$ [ppb]')
axs[1,1].set_xlabel('CH$_4$ [ppb]')
axs[0,0].set_ylabel("Pressure [hPa]")
axs[1,0].set_ylabel("Pressure [hPa]")
axs[0,0].set_ylim(1050, -20) 
axs[1,0].set_ylim(1050, -20) 

plt.tight_layout()
plt.savefig(save_dir + 'Fig_3.png', bbox_inches='tight', transparent=True)

# %% Figure 4. TCCON vs. JR-Station trend and seasonality

fig, axs = plt.subplots(2,2,figsize=(18*ic_cm,15*ic_cm*.8), dpi=300,
                        sharex='col', gridspec_kw={'width_ratios': [1.5, 1]})

# Trend
# axs[0].plot(tccon_fit_eu.Date, tccon_fit_eu.trend_co2, label='Eureka 80°N', lw=.4)
axs[0,0].plot(tccon_fit_so.Date, tccon_fit_so.trend_co2, c='darkcyan', label='67°N Sodankylä', lw=.4)
axs[0,0].plot(long_co2_fit_dem.Date, long_co2_fit_dem.trend, c='steelblue',
              linestyle='dashed', label='60°N DEM', lw=.8)
axs[0,0].plot(long_co2_fit_brz.Date, long_co2_fit_brz.trend, c="teal",
              linestyle='dotted', label='56°N BRZ', lw=.8)
axs[0,0].plot(tccon_fit_es.Date, tccon_fit_es.trend_co2, c='olivedrab', label='54°N East Trout Lake', lw=.4)
axs[0,0].plot(tccon_fit_br.Date, tccon_fit_br.trend_co2, c='orange', label='53°N Bremen', lw=.4)
axs[0,0].plot(tccon_fit_bi.Date, tccon_fit_bi.trend_co2, c='gold', label='53°N Bialystok', lw=.4)
axs[0,0].plot(tccon_fit_pa.Date, tccon_fit_pa.trend_co2, c='darkgreen', label='46°N Park Falls ', lw=.4)

# axs[1].plot(tccon_fit_eu.Date, tccon_fit_eu.trend_ch4, label='Eureka 80°N', lw=.4)
# axs[1].plot(tccon_fit_ny.Date, tccon_fit_ny.trend_ch4, label='Ny-Ålesund 79°N', lw=.4)
axs[1,0].plot(tccon_fit_so.Date, tccon_fit_so.trend_ch4, c='darkcyan',label='Sodankylä 67°N', lw=.4)
axs[1,0].plot(long_ch4_fit_dem.Date, long_ch4_fit_dem.trend, c='steelblue',
              linestyle='dashed', label='DEM 60°N', lw=.8)
axs[1,0].plot(long_ch4_fit_brz.Date, long_ch4_fit_brz.trend, c="teal",
              linestyle='dotted', label='brz 56°N', lw=.8)
axs[1,0].plot(tccon_fit_es.Date, tccon_fit_es.trend_ch4, c='olivedrab', label='East Trout Lake 54°N', lw=.4)
axs[1,0].plot(tccon_fit_br.Date, tccon_fit_br.trend_ch4, c='orange', label='Bremen 53°N', lw=.4)
axs[1,0].plot(tccon_fit_bi.Date, tccon_fit_bi.trend_ch4, c='gold', label='Bialystok 53°N', lw=.4)
axs[1,0].plot(tccon_fit_pa.Date, tccon_fit_pa.trend_ch4, c='darkgreen', label='Park Falls 46°N', lw=.4)

# Seasonality
axs[0,1].plot(tccon_fit_so.Date, tccon_fit_so.harmonics_co2, c='darkcyan',label='Sodankylä 67°N', lw=.4)
axs[0,1].plot(long_co2_fit_dem.Date, long_co2_fit_dem.harmonics, c='steelblue',
              linestyle='dashed', label='DEM 60°N', lw=.8)
axs[0,1].plot(long_co2_fit_brz.Date, long_co2_fit_brz.harmonics, c="teal",
              linestyle='dotted', label='BRZ 56°N', lw=.8)
axs[0,1].plot(tccon_fit_es.Date, tccon_fit_es.harmonics_co2, c='olivedrab', label='East Trout Lake 54°N', lw=.4)
axs[0,1].plot(tccon_fit_br.Date, tccon_fit_br.harmonics_co2, c='orange', label='Bremen 53°N', lw=.4)
axs[0,1].plot(tccon_fit_bi.Date, tccon_fit_bi.harmonics_co2, c='gold', label='Bialystok 53°N', lw=.4)
axs[0,1].plot(tccon_fit_pa.Date, tccon_fit_pa.harmonics_co2, c='darkgreen', label='Park Falls 46°N', lw=.4)
# axs[0].plot(gosat_co2_detrend_BRZ.Date, gosat_co2_detrend_BRZ.harmonics, label='BRZ-GOSAT 56°N', lw=.4)
# axs[0].plot(gosat_co2_detrend_DEM.Date, gosat_co2_detrend_DEM.harmonics, label='DEM-GOSAT 60°N', lw=.4)

# axs[1].plot(tccon_fit_eu.Date, tccon_fit_eu.harmonics_ch4, label='Eureka 80°N', lw=.4)
# axs[1].plot(tccon_fit_ny.Date, tccon_fit_ny.harmonics_ch4, label='Ny-Ålesund 80°N', lw=.4)
axs[1,1].plot(tccon_fit_so.Date, tccon_fit_so.harmonics_ch4, c='darkcyan',label='Sodankylä 67°N', lw=.4)
axs[1,1].plot(long_ch4_fit_dem.Date, long_ch4_fit_dem.harmonics, c='steelblue',
              linestyle='dashed', label='DEM 60°N', lw=.8)
axs[1,1].plot(long_ch4_fit_brz.Date, long_ch4_fit_brz.harmonics, c="teal",
              linestyle='dotted', label='BRZ 56°N', lw=.8)
axs[1,1].plot(tccon_fit_es.Date, tccon_fit_es.harmonics_ch4, c='olivedrab', label='East Trout Lake 54°N', lw=.4)
axs[1,1].plot(tccon_fit_br.Date, tccon_fit_br.harmonics_ch4, c='orange', label='Bremen 53°N', lw=.4)
axs[1,1].plot(tccon_fit_bi.Date, tccon_fit_bi.harmonics_ch4, c='gold', label='Bialystok 53°N', lw=.4)
axs[1,1].plot(tccon_fit_pa.Date, tccon_fit_pa.harmonics_ch4, c='darkgreen', label='Park Falls 46°N', lw=.4)

axs[0,0].set_ylabel('XCO$_2$ [ppm]')
axs[1,0].set_ylabel('XCH$_4$ [ppb]')
axs[0,1].set_ylabel('XCO$_2$ anomaly [ppm]')
axs[1,1].set_ylabel('XCH$_4$ anomaly [ppb]')
axs[1,0].set_xlabel('Year')
axs[1,1].set_xlabel('Time')
axs[1,1].set_xlim(datetime(2017,1,1), datetime(2018,1,1))
axs[1,1].set_xticks(axs[1,1].get_xticks()[::2])
axs[0,0].text(datetime(2024,12,1), 425, "(a)", fontsize=10)
axs[1,0].text(datetime(2024,12,1), 1915, "(c)", fontsize=10)
axs[0,1].text(datetime(2017,11,20), 4, "(b)", fontsize=10)
axs[1,1].text(datetime(2017,11,20), 14, "(d)", fontsize=10)
axs[0,0].legend(ncol=1, fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)

plt.savefig(save_dir + 'Fig_4.png', bbox_inches='tight', transparent=True)

# %% For reference: TCCON vs. JR-Station detrended

fig, axs = plt.subplots(2,1,figsize=(18*ic_cm,15*ic_cm*.8), dpi=300, sharex='col')

# axs[0].plot(tccon_res_eu.Date, tccon_res_eu.res_co2, label='Eureka 80°N', lw=.4)
# axs[0].plot(tccon_res_ny.Date, tccon_res_ny.res_co2, label='Ny-Ålesund 79°N', lw=.4)
axs[0].plot(tccon_res_so.Date, tccon_res_so.detrend_co2, label='Sodankylä 67°N', lw=.4)
axs[0].plot(long_co2_res_dem.Date, long_co2_res_dem.detrend, label='DEM 60°N', lw=.4)
axs[0].plot(long_co2_res_brz.Date, long_co2_res_brz.detrend, label='BRZ 56°N', lw=.4)
axs[0].plot(tccon_res_es.Date, tccon_res_es.detrend_co2, label='East Trout Lake 54°N', lw=.4)
axs[0].plot(tccon_res_br.Date, tccon_res_br.detrend_co2, label='Bremen 53°N', lw=.4)
axs[0].plot(tccon_res_bi.Date, tccon_res_bi.detrend_co2, label='Bialystok 53°N', lw=.4)
axs[0].plot(tccon_res_pa.Date, tccon_res_pa.detrend_co2, label='Park Falls 46°N', lw=.4)

# axs[1].plot(tccon_res_ny.Date, tccon_res_ny.res_ch4, label='Ny-Ålesund 80°N', lw=.4)
# axs[1].plot(tccon_res_eu.Date, tccon_res_eu.res_ch4, label='Eureka 79°N', lw=.4)
axs[1].plot(tccon_res_so.Date, tccon_res_so.detrend_ch4, label='Sodankylä 67°N', lw=.4)
axs[1].plot(long_ch4_res_dem.Date, long_ch4_res_dem.detrend, label='DEM 60°N', lw=.4)
axs[1].plot(long_ch4_res_brz.Date, long_ch4_res_brz.detrend, label='BRZ 56°N', lw=.4)
axs[1].plot(tccon_res_es.Date, tccon_res_es.detrend_ch4, label='East Trout Lake 54°N', lw=.4)
axs[1].plot(tccon_res_br.Date, tccon_res_br.detrend_ch4, label='Bremen 53°N', lw=.4)
axs[1].plot(tccon_res_bi.Date, tccon_res_bi.detrend_ch4, label='Bialystok 53°N', lw=.4)
axs[1].plot(tccon_res_pa.Date, tccon_res_pa.detrend_ch4, label='Park Falls 46°N', lw=.4)

axs[0].set_ylabel('XCO$_2$ [ppm]')
axs[1].set_ylabel('XCH$_4$ [ppb]')
axs[1].set_xlabel('Year')
axs[1].legend(ncol=1, fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)

# %% Figure 6, 3 types XCH4

# Define gas configuration
gas_cfg = {
    'name': r'CH_{4}', 
    'unit': 'ppm', 
    'col_key': 'CH4_cak', 
    'gosat_key': 'CH4_gosat',
    'gas_key': 'xch4_gosat'
}

fig, axs = plt.subplots(2, 2, figsize=(18*ic_cm*.9, 15*ic_cm*.85), dpi=300, 
                        sharex='col', sharey='col', gridspec_kw={'width_ratios': [2.5, 1]})

ana.plot_timeseries_panel(axs[0,0], gosat_ch4_dem, norm_ch4_dem, out_ch4_dem,
                          reg_ch4_dem, anom_ch4_dem, long_ch4_fit_dem, "DEM", "(a)", gas_cfg)
ana.plot_seasonal_panel(axs[0,1], norm_ch4_dem, reg_ch4_dem, compare_gosat_ch4_dem, "(b)", gas_cfg, 9.8, 8.6)

ana.plot_timeseries_panel(axs[1,0], gosat_ch4_brz, norm_ch4_brz, out_ch4_brz,
                          reg_ch4_brz, anom_ch4_brz, long_ch4_fit_brz, "BRZ", "(c)", gas_cfg)
ana.plot_seasonal_panel(axs[1,1], norm_ch4_brz, reg_ch4_brz, compare_gosat_ch4_brz, "(d)", gas_cfg, -9.5, -10.7)

# --- Final Formatting ---
names = ['MAM','JJA','SON','DNF']
handles, labels = axs[1,0].get_legend_handles_labels()
order = [4,5,3,6,0,2,1]
axs[1,0].legend([handles[i] for i in order], [labels[i] for i in order],
                bbox_to_anchor=(0.44, 0.7), loc='center right',
                ncol=2, fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)
# axs[1,1].legend(ncol=1, loc='lower right', fontsize=6, handletextpad=0.1, columnspacing=0.1, markerscale=.5, borderaxespad=1)
axs[1,0].set_xlabel('Date')
axs[1,1].set_xlabel('Month')
axs[0,0].set_ylabel('XCH$_4$ [ppb]')
axs[0,1].set_ylabel(r'$\Delta$XCH$_4$ [ppb]')
axs[1,0].set_ylabel('XCH$_4$ [ppb]')
axs[1,1].set_ylabel(r'$\Delta$XCH$_4$ [ppb]')
axs[0,1].set_xlim(.5,4.5)
# axs[0,1].set_ylim(-20,5)
# axs[1,1].set_ylim(-5,5)
axs[1,1].set_xticks([1,2,3,4],names)
axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Legends
h, l = axs[1,0].get_legend_handles_labels()
# axs[0,0].legend([h[i] for i in [4,5,3,6,0,2,1]], [l[i] for i in [4,5,3,6,0,2,1]],
#                 ncol=2, fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)
axs[1,1].legend(ncol=1, loc='lower left', fontsize=6, handletextpad=0.05, columnspacing=0.1, markerscale=.5, borderaxespad=1)

plt.tight_layout()
plt.savefig(save_dir + 'Fig_6.png', bbox_inches='tight', transparent=True)

# %% Figure 5, 3 types XCO2

gas_cfg = {
    'name': r'CO_{2}', 
    'unit': 'ppm', 
    'col_key': 'CO2_cak', 
    'gosat_key': 'CO2_gosat',
    'gas_key': 'xco2_gosat'
}

fig, axs = plt.subplots(2, 2, figsize=(18*ic_cm*.9, 15*ic_cm*.85), dpi=300, 
                        sharex='col', sharey='col', gridspec_kw={'width_ratios': [2.5, 1]})

ana.plot_timeseries_panel(axs[0,0], gosat_co2_dem, norm_co2_dem, out_co2_dem,
                          reg_co2_dem, anom_co2_dem, long_co2_fit_dem, "DEM", "(a)", gas_cfg)
ana.plot_seasonal_panel(axs[0,1], norm_co2_dem, reg_co2_dem, compare_gosat_co2_dem, "(b)", gas_cfg, 3, 2.7)

ana.plot_timeseries_panel(axs[1,0], gosat_co2_brz, norm_co2_brz, out_co2_brz,
                          reg_co2_brz, anom_co2_brz, long_co2_fit_brz, "BRZ", "(c)", gas_cfg)
ana.plot_seasonal_panel(axs[1,1], norm_co2_brz, reg_co2_brz, compare_gosat_co2_brz, "(d)", gas_cfg, 3, 2.7)

# Formatting
axs[0,1].set_xlim(.5, 4.5)
axs[0,1].set_ylim(-3.5, 3.5)
axs[0,0].set_ylim(360, 440)
axs[1,1].set_xticks([1,2,3,4], ['MAM','JJA','SON','DNF'])
axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Legends
h, l = axs[1,0].get_legend_handles_labels()
axs[0,0].legend([h[i] for i in [4,5,3,6,0,2,1]], [l[i] for i in [4,5,3,6,0,2,1]],
                bbox_to_anchor=(0.01, 0.7), loc='center left',
                ncol=2, fontsize=6, handletextpad=0.05, columnspacing=0.1, borderaxespad=1)
axs[0,1].legend(ncol=1, loc='lower left', fontsize=6, handletextpad=0.05, columnspacing=0.1, markerscale=.5, borderaxespad=1)
axs[1,0].set_xlabel('Date')
axs[1,1].set_xlabel('Month')
axs[0,0].set_ylabel('XCO$_2$ [ppm]')
axs[0,1].set_ylabel(r'$\Delta$XCO$_2$ [ppm]')
axs[1,0].set_ylabel('XCO$_2$ [ppm]')
axs[1,1].set_ylabel(r'$\Delta$XCO$_2$ [ppm]')

plt.tight_layout()
plt.savefig(save_dir + 'Fig_5.png', bbox_inches='tight', transparent=True)

# %% For reference: CO2 auxiliary parameters
fig, axs = plt.subplots(3,5,figsize=(18*ic_cm,15*ic_cm), sharey='all', sharex='col', dpi=300, constrained_layout=True)

# Determine the global min and max of AOT across both sites
global_aot_min = min(compare_gosat_co2_dem.snrco2w.min(), compare_gosat_co2_brz.snrco2w.min())
global_aot_max = max(compare_gosat_co2_dem.snrco2w.max(), compare_gosat_co2_brz.snrco2w.max())

# Create a unified normalization object
shared_norm = mcolors.Normalize(vmin=global_aot_min, vmax=global_aot_max)

axs[0,0].scatter(compare_gosat_co2_dem.sanir, compare_gosat_co2_dem.Diff,
                 c=compare_gosat_co2_dem.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[0,1].scatter([np.sum(x[-4:-1]) for x in compare_gosat_co2_dem.layer_se], compare_gosat_co2_dem.Diff,
                 c=compare_gosat_co2_dem.snrco2w,
                 cmap='Blues',  norm=shared_norm,s=3.5)
axs[0,2].scatter([np.sum(x[:3]) for x in compare_gosat_co2_dem.layer_se], compare_gosat_co2_dem.Diff,
                 c=compare_gosat_co2_dem.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[0,3].scatter(compare_gosat_co2_dem.aot, compare_gosat_co2_dem.Diff,
                 c=compare_gosat_co2_dem.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
sc1 = axs[0,4].scatter(
    compare_gosat_co2_dem.sza, compare_gosat_co2_dem.Diff, c=compare_gosat_co2_dem.snrco2w,
    cmap='Blues', norm=shared_norm, s=3.5
)
cbar1 = fig.colorbar(sc1, ax=axs[0, :], orientation='vertical', shrink=0.8, pad=0.01)
cbar1.set_label('CO$_2$ SNR')
axs[0,0].set_ylabel(r'$X_{\mathrm{CO}_2}^{\mathrm{GOSAT}} - X_{\mathrm{CO}_2}^{\mathrm{CAK}}$ [ppm]')

axs[1,0].scatter(compare_gosat_co2_brz.sanir, compare_gosat_co2_brz.Diff,
                 c=compare_gosat_co2_brz.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[1,1].scatter([np.sum(x[-4:-1]) for x in compare_gosat_co2_brz.layer_se], compare_gosat_co2_brz.Diff,
                 c=compare_gosat_co2_brz.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[1,2].scatter([np.sum(x[:3]) for x in compare_gosat_co2_brz.layer_se], compare_gosat_co2_brz.Diff,
                 c=compare_gosat_co2_brz.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[1,3].scatter(compare_gosat_co2_brz.aot, compare_gosat_co2_brz.Diff,
                 c=compare_gosat_co2_brz.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
sc2 = axs[1,4].scatter(compare_gosat_co2_brz.sza, compare_gosat_co2_brz.Diff,
                       c=compare_gosat_co2_brz.snrco2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
cbar2 = fig.colorbar(sc2, ax=axs[1, :], orientation='vertical', shrink=0.8, pad=0.01)
cbar2.set_label('CO$_2$ SNR')

axs[2,0].scatter(compare_co2_so.SA_NIR, compare_co2_so.Diff, c=compare_co2_so.SNR_CO2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
axs[2,3].scatter(compare_co2_so.AOT, compare_co2_so.Diff, c=compare_co2_so.SNR_CO2w,
                 cmap='Blues', norm=shared_norm, s=3.5)
sc3 = axs[2,4].scatter(compare_co2_so.SZA, compare_co2_so.Diff,
                       c=compare_co2_so.SNR_CO2w,
                       cmap='Blues', norm=shared_norm, s=3.5)
cbar3 = fig.colorbar(sc3, ax=axs[2, :], orientation='vertical', shrink=0.8, pad=0.01)
cbar3.set_label('CO$_2$ SNR')

s_name = ['MAM','JJA','SON','DJF']
colors = [ 'darkgreen',  'firebrick',  'goldenrod', 'steelblue']

for i in [1,2,3]:
    group = compare_gosat_co2_dem[compare_gosat_co2_dem['Season'] == i]
    group_sse = np.array([np.sum(x[-4:-1]) for x in group.layer_se])
    group_stse = np.array([np.sum(x[:4]) for x in group.layer_se])
    
    # Plot the Confidence Ellipse (e.g., 2 standard deviations)
    ana.draw_confidence_ellipse(group['sa'], group['Diff'], axs[0,0], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,0].scatter(group['sa'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group_sse, group['Diff'], axs[0,1], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,1].scatter(group_sse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group_stse, group['Diff'], axs[0,2], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,2].scatter(group_stse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['aot'], group['Diff'], axs[0,3], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,3].scatter(group['aot'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['sza'], group['Diff'], axs[0,4], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,4].scatter(group['sza'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    
for i in [1,2,3]:
    group = compare_gosat_co2_brz[compare_gosat_co2_brz['Season'] == i]
    group_sse = np.array([np.sum(x[-4:-1]) for x in group.layer_se])
    group_stse = np.array([np.sum(x[:4]) for x in group.layer_se])
    
    # Plot the Confidence Ellipse (e.g., 2 standard deviations)
    ana.draw_confidence_ellipse(group['sa'], group['Diff'], axs[1,0], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,0].scatter(group['sa'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group_sse, group['Diff'], axs[1,1], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,1].scatter(group_sse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')        
    ana.draw_confidence_ellipse(group_stse, group['Diff'], axs[1,2], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])    
    axs[1,2].scatter(group_stse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['aot'], group['Diff'], axs[1,3], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,3].scatter(group['aot'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['sza'], group['Diff'], axs[1,4], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,4].scatter(group['sza'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')

axs[1,4].legend(loc='best', fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)
axs[1,0].set_xlabel('Surface Albedo')
axs[1,0].set_ylabel(r'$X_{\mathrm{CO}_2}^{\mathrm{GOSAT}} - X_{\mathrm{CO}_2}^{\mathrm{CAK}}$ [ppm]')
# axs[1,1].set_xlabel(r'$\Delta$P$_0$ [hPa]')
axs[1,1].set_xlabel('Surface SE [ppm]')
axs[1,2].set_xlabel('Strat. SE [ppm]')
axs[1,3].set_xlabel('AOT')
axs[1,4].set_xlabel('SZA [°]')

axs[0,0].text(0.3, 0.96, 'DEM', transform=axs[0,0].transAxes, fontsize=10, ha='right', va='top')
axs[1,0].text(0.3, 0.96, 'BRZ', transform=axs[1,0].transAxes, fontsize=10, ha='right', va='top')

# %% For reference: CH4 auxiliary parameters
fig, axs = plt.subplots(2,5,figsize=(18*ic_cm,15*ic_cm*.7), sharey='all', sharex='col', dpi=300, constrained_layout=True)
s_name = ['MAM', 'JJA', 'SON']

# 1. Determine the global min and max of AOT across both sites
global_aot_min = min(compare_gosat_ch4_dem.snrch4.min(), compare_gosat_ch4_brz.snrch4.min())
global_aot_max = max(compare_gosat_ch4_dem.snrch4.max(), compare_gosat_ch4_brz.snrch4.max())

# 2. Create a unified normalization object
shared_norm = mcolors.Normalize(vmin=global_aot_min, vmax=global_aot_max)

axs[0,0].scatter(compare_gosat_ch4_dem.sanir/compare_gosat_ch4_dem.saswir,
                 compare_gosat_ch4_dem.Diff, c=compare_gosat_ch4_dem.snrch4,
                 cmap='Purples', norm=shared_norm, s=2.5)
axs[0,1].scatter([x[-1] for x in compare_gosat_ch4_dem.layer_se], compare_gosat_ch4_dem.Diff, c=compare_gosat_ch4_dem.snrch4,
                 cmap='Purples', norm=shared_norm, s=2.5)
axs[0,2].scatter(compare_gosat_ch4_dem.total_se, compare_gosat_ch4_dem.Diff, c=compare_gosat_ch4_dem.snrch4,
                 cmap='Purples', norm=shared_norm, s=2.5)
axs[0,3].scatter(compare_gosat_ch4_dem.aot, compare_gosat_ch4_dem.Diff, c=compare_gosat_ch4_dem.snrch4,
                 cmap='Purples', norm=shared_norm, s=2.5)
sc1 = axs[0,4].scatter(
    compare_gosat_ch4_dem.sza, compare_gosat_ch4_dem.Diff, c=compare_gosat_ch4_dem.snrch4,
    cmap='Purples', norm=shared_norm, s=2.5)

cbar1 = fig.colorbar(sc1, ax=axs[0, :], orientation='vertical', shrink=0.8, pad=0.01)
cbar1.set_label('CH$_4$ SNR')

axs[1,0].scatter(compare_gosat_ch4_brz.sanir/compare_gosat_ch4_brz.saswir,
                 compare_gosat_ch4_brz.Diff, c=compare_gosat_ch4_brz.snrch4,
                 cmap='Purples', norm=shared_norm, alpha=.7, s=2.5)
axs[1,1].scatter([x[-1] for x in compare_gosat_ch4_brz.layer_se], compare_gosat_ch4_brz.Diff, c=compare_gosat_ch4_brz.snrch4,
                 cmap='Purples', norm=shared_norm, alpha=.7, s=2.5)
axs[1,2].scatter(compare_gosat_ch4_brz.total_se, compare_gosat_ch4_brz.Diff, c=compare_gosat_ch4_brz.snrch4,
                 cmap='Purples', norm=shared_norm, alpha=.7, s=2.5)
axs[1,3].scatter(compare_gosat_ch4_brz.aot, compare_gosat_ch4_brz.Diff, c=compare_gosat_ch4_brz.snrch4,
                 cmap='Purples', norm=shared_norm, alpha=.7, s=2.5)
sc2 = axs[1,4].scatter(
    compare_gosat_ch4_brz.sza, compare_gosat_ch4_brz.Diff, c=compare_gosat_ch4_brz.snrch4,
    cmap='Purples', norm=shared_norm, alpha=.7, s=2.5
)

cbar2 = fig.colorbar(sc2, ax=axs[1, :], orientation='vertical', shrink=0.8, pad=0.01)
cbar2.set_label('CH$_4$ SNR')

for i in [1,2,3]:
    group = compare_gosat_ch4_dem[compare_gosat_ch4_dem['Season'] == i]
    group_sse = np.array([x[-1] for x in group.layer_se])
    
    # Plot the Confidence Ellipse (e.g., 2 standard deviations)
    ana.draw_confidence_ellipse(group['sao2'], group['Diff'], axs[0,0], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,0].scatter(group['sao2'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group_sse, group['Diff'], axs[0,1], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,1].scatter(group_sse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    axs[0,2].scatter(group['total_se'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['total_se'], group['Diff'], axs[0,2], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,3].scatter(group['aot'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['aot'], group['Diff'], axs[0,3], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    ana.draw_confidence_ellipse(group['sza'], group['Diff'], axs[0,4], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[0,4].scatter(group['sza'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')


for i in [1,2,3]:
    group = compare_gosat_ch4_brz[compare_gosat_ch4_brz['Season'] == i]
    group_sse = np.array([x[-1] for x in group.layer_se])
    
    # Plot the Confidence Ellipse (e.g., 2 standard deviations)
    ana.draw_confidence_ellipse(group['sao2'], group['Diff'], axs[1,0], n_std=2.0, 
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,0].scatter(group['sao2'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group_sse, group['Diff'], axs[1,1], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,1].scatter(group_sse.mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['total_se'], group['Diff'], axs[1,2], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,2].scatter(group['total_se'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['aot'], group['Diff'], axs[1,3], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,3].scatter(group['aot'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')
    ana.draw_confidence_ellipse(group['sza'], group['Diff'], axs[1,4], n_std=2.0,
                                edgecolor=colors[i-1], lw=.5, label=s_name[i-1])
    axs[1,4].scatter(group['sza'].mean(), group['Diff'].mean(), color=colors[i-1], marker='x')

axs[1,4].legend(loc='best', fontsize=6, handletextpad=0.1, columnspacing=0.1, borderaxespad=1)
axs[1,0].set_xlabel('Surface Albedo')
axs[0,0].set_ylabel(r'$X_{\mathrm{CH}_4}^{\mathrm{GOSAT}} - X_{\mathrm{CH}_4}^{\mathrm{CAK}}$ [ppb]')
axs[1,0].set_ylabel(r'$X_{\mathrm{CH}_4}^{\mathrm{GOSAT}} - X_{\mathrm{CH}_4}^{\mathrm{CAK}}$ [ppb]')
# axs[1,1].set_xlabel(r'$\Delta$P$_0$ [hPa]')
axs[1,1].set_xlabel('Sufrase SE [ppb]')
axs[1,2].set_xlabel('Total SE [ppb]')
axs[1,3].set_xlabel('AOT')
axs[1,4].set_xlabel('SZA [°]')

axs[0,0].text(0.3, 0.96, 'DEM', transform=axs[0,0].transAxes, fontsize=10, ha='right', va='top')
axs[1,0].text(0.3, 0.96, 'BRZ', transform=axs[1,0].transAxes, fontsize=10, ha='right', va='top')

# %% For reference: Total profiles ploted, DEM CO
fig, axs = plt.subplots(3, 6, figsize=(9,7), sharey = 'all', sharex = 'row', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

years = column_long_co_dem.Year.unique()
x = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2]
y = [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5]

for year in range(len(years)):
    # test = column_tp_adj2_DEM.loc[column_tp_adj2_DEM.Year == years[year]]
    test = column_long_co_dem.loc[column_long_co_dem.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')

    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CO_column, test.iloc[i].Pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        axs[x[year],y[year]].scatter(test.iloc[i].CO_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        
axs[2,0].set_xlabel('CO [ppb]')
axs[2,1].set_xlabel('CO [ppb]')
axs[2,2].set_xlabel('CO [ppb]')
axs[2,3].set_xlabel('CO [ppb]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, -10) 
# axs[0,0].set_xlim(1800, 2300)

fig.suptitle("Profiles of DEM (60°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()

# %% For reference: Total profiles ploted, DEM CO2 

fig, axs = plt.subplots(3, 4, figsize=(9,7), sharey = 'all', sharex = 'row', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

years = column_compare_co2_dem.Year.unique()
x = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
y = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

for year in range(len(years)):
    test = column_compare_co2_dem.loc[column_compare_co2_dem.Year == years[year]]
    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CO2_camsinv, test.iloc[i].Pressure_camsinv,
                                      c='grey', alpha=.3, linewidth = .7)
for year in range(len(years)):
    test = column_compare_co2_dem.loc[column_compare_co2_dem.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')

    for i in range(len(test)):    
        co2_column = np.append(test.iloc[i].CO2_2, test.iloc[i].CO2_1)
        pressure_column = np.append(test.iloc[i].Pressure_2, test.iloc[i].Pressure_1)
        # sort to upward
        sort_indices = np.argsort(pressure_column)
        sorted_pressure_column = pressure_column[sort_indices][::-1]
        sorted_co2_column = co2_column[sort_indices][::-1]
        
        axs[x[year],y[year]].plot(sorted_co2_column, sorted_pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        axs[x[year],y[year]].scatter(test.iloc[i].CO2_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        axs[x[year],y[year]].scatter(test.iloc[i].CO2_tower, test.iloc[i].Pressure_tower, marker='2',
                                     c=colors[int(test.iloc[i].Season-1)], s=5, alpha=.8)
        
axs[2,0].set_xlabel('CO$_2$ [ppm]')
axs[2,1].set_xlabel('CO$_2$ [ppm]')
axs[2,2].set_xlabel('CO$_2$ [ppm]')
axs[2,3].set_xlabel('CO$_2$ [ppm]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, -10) 
# axs[0,0].set_xlim(1800, 2300)

fig.suptitle("Profiles of DEM (60°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()
# %% For reference: Troposphere profiles ploted, DEM CH4
fig, axs = plt.subplots(3, 4, figsize=(9,7), sharey = 'all', sharex = 'all', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

# years = column_tp_adj2_DEM.Year.unique()
years = column_compare_ch4_dem.Year.unique()
x = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
y = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

for year in range(len(years)):
    # test = column_tp_adj2_DEM.loc[column_tp_adj2_DEM.Year == years[year]]    
    test = column_compare_ch4_dem.loc[column_compare_ch4_dem.Year == years[year]]
    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CH4_camsinv, test.iloc[i].Pressure_camsinv,
                                      c='grey', alpha=.3, linewidth = .7)
for year in range(len(years)):
    # test = column_tp_adj2_DEM.loc[column_tp_adj2_DEM.Year == years[year]]
    test = column_compare_ch4_dem.loc[column_compare_ch4_dem.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')

    for i in range(len(test)):    
        ch4_column = np.append(test.iloc[i].CH4_2, np.append(test.iloc[i].CH4_1, test.iloc[i].CH4_0))
        pressure_column = np.append(test.iloc[i].Pressure_2, np.append(test.iloc[i].Pressure_1, test.iloc[i].Pressure_0))
        # sort to upward
        sort_indices = np.argsort(pressure_column)
        sorted_pressure_column = pressure_column[sort_indices][::-1]
        sorted_ch4_column = ch4_column[sort_indices][::-1]
        
        axs[x[year],y[year]].plot(sorted_ch4_column, sorted_pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        axs[x[year],y[year]].scatter(test.iloc[i].CH4_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        axs[x[year],y[year]].scatter(test.iloc[i].CH4_tower, test.iloc[i].Pressure_tower, marker='2',
                                     c=colors[int(test.iloc[i].Season-1)], s=5, alpha=.8)
        
axs[2,0].set_xlabel('CH$_4$ [ppb]')
axs[2,1].set_xlabel('CH$_4$ [ppb]')
axs[2,2].set_xlabel('CH$_4$ [ppb]')
axs[2,3].set_xlabel('CH$_4$ [ppb]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, 100) 
axs[0,0].set_xlim(1800, 2200)

fig.suptitle("Profiles of DEM (60°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()

# %% For reference: Total profiles ploted, BRZ CO
fig, axs = plt.subplots(4, 6, figsize=(9,7), sharey = 'all', sharex = 'row', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

years = column_long_co_brz.Year.unique()
x = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
y = [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5]

for year in range(len(years)):
    test = column_long_co_brz.loc[column_long_co_brz.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')

    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CO_column, test.iloc[i].Pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        axs[x[year],y[year]].scatter(test.iloc[i].CO_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        
axs[3,0].set_xlabel('CO [ppb]')
axs[3,1].set_xlabel('CO [ppb]')
axs[3,2].set_xlabel('CO [ppb]')
axs[3,3].set_xlabel('CO [ppb]')
axs[3,4].set_xlabel('CO [ppb]')
axs[3,5].set_xlabel('CO [ppb]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[3,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, -10) 
# axs[0,0].set_xlim(1800, 2300)

fig.suptitle("Profiles of BRZ (56°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()


# %% For reference: Total profiles ploted, BRZ CO2
fig, axs = plt.subplots(3, 5, figsize=(9,7), sharey = 'all', sharex = 'row', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

# years = column_tp_adj2_DEM.Year.unique()
years = column_compare_co2_brz.Year.unique()
x = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
y = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

for year in range(len(years)):  
    test = column_compare_co2_brz.loc[column_compare_co2_brz.Year == years[year]]
    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CO2_camsinv, test.iloc[i].Pressure_camsinv,
                                      c='grey', alpha=.3, linewidth = .7)
for year in range(len(years)):
    # test = column_tp_adj2_DEM.loc[column_tp_adj2_DEM.Year == years[year]]
    test = column_compare_co2_brz.loc[column_compare_co2_brz.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')

    for i in range(len(test)):    
        co2_column = np.append(test.iloc[i].CO2_2, test.iloc[i].CO2_1)
        pressure_column = np.append(test.iloc[i].Pressure_2, test.iloc[i].Pressure_1)
        # sort to upward
        sort_indices = np.argsort(pressure_column)
        sorted_pressure_column = pressure_column[sort_indices][::-1]
        sorted_co2_column = co2_column[sort_indices][::-1]
        
        axs[x[year],y[year]].plot(sorted_co2_column, sorted_pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        axs[x[year],y[year]].scatter(test.iloc[i].CO2_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        axs[x[year],y[year]].scatter(test.iloc[i].CO2_tower, test.iloc[i].Pressure_tower, marker='2',
                                     c=colors[int(test.iloc[i].Season-1)], s=5, alpha=.8)
        
axs[2,0].set_xlabel('CO$_2$ [ppm]')
axs[2,1].set_xlabel('CO$_2$ [ppm]')
axs[2,2].set_xlabel('CO$_2$ [ppm]')
axs[2,3].set_xlabel('CO$_2$ [ppm]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, -10) 
# axs[0,0].set_xlim(1800, 2300)

fig.suptitle("Profiles of BRZ (56°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()

# %% For reference: Troposphere profiles ploted, BRZ CH4
fig, axs = plt.subplots(3, 5, figsize=(10,7.5), sharey = 'all', sharex = 'all', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

years = column_compare_ch4_brz.Year.unique()
x = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
y = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

for year in range(len(years)):
    test = column_compare_ch4_brz.loc[column_compare_ch4_brz.Year == years[year]]
    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].CH4_camsinv, test.iloc[i].Pressure_camsinv,
                                      c='grey', alpha=.3, linewidth = .7)
for year in range(len(years)):
    test = column_compare_ch4_brz.loc[column_compare_ch4_brz.Year == years[year]]
    axs[x[year],y[year]].set_title(str(int(years[year])), fontsize=10, fontweight='bold')
    for i in range(len(test)):
        ch4_column = np.append(test.iloc[i].CH4_2, np.append(test.iloc[i].CH4_1, test.iloc[i].CH4_0))
        pressure_column = np.append(test.iloc[i].Pressure_2, np.append(test.iloc[i].Pressure_1, test.iloc[i].Pressure_0))
        # sort to upward
        sort_indices = np.argsort(pressure_column)
        sorted_pressure_column = pressure_column[sort_indices][::-1]
        sorted_ch4_column = ch4_column[sort_indices][::-1]
        
        axs[x[year],y[year]].plot(sorted_ch4_column, sorted_pressure_column, linewidth = 1, alpha=.7,
                                  c=colors[int(test.iloc[i].Season-1)])
        
        axs[x[year],y[year]].scatter(test.iloc[i].CH4_aircraft, test.iloc[i].Pressure_aircraft,
                                     c=colors[int(test.iloc[i].Season-1)], s=1.5, alpha=.6)
        axs[x[year],y[year]].scatter(test.iloc[i].CH4_tower, test.iloc[i].Pressure_tower, marker='2',
                                     c=colors[int(test.iloc[i].Season-1)], s=5, alpha=.8)
        # axs[x[year],y[year]].axhline(y=test.iloc[i].Pressure_PBL, color='grey', linestyle='--', alpha=.5)
        
axs[2,0].set_xlabel('CH$_4$ [ppb]')
axs[2,1].set_xlabel('CH$_4$ [ppb]')
axs[2,2].set_xlabel('CH$_4$ [ppb]')
axs[2,3].set_xlabel('CH$_4$ [ppb]')
axs[2,4].set_xlabel('CH$_4$ [ppb]')
axs[0,0].set_ylabel('Pressure [hPa]')
axs[1,0].set_ylabel('Pressure [hPa]')
axs[2,0].set_ylabel('Pressure [hPa]')
axs[0,0].set_ylim(1050, -10) 
axs[0,0].set_xlim(1800, 2200)

fig.suptitle("Profiles of BRZ (56°N), 2009 to 2021", weight = 'semibold', fontsize=12)
plt.tight_layout()

# %% For reference: GOSAT vs. TCCON & JR-Station, box plot

fig, axs = plt.subplots(2,1, figsize=(18*ic_cm,15*ic_cm), dpi=300, sharex='col')

# XCO2
axs[0].errorbar(np.array(compare_co2_so[2]) - 0.15, compare_co2_so[0], 
                yerr=compare_co2_so[1], fmt='o', markersize=3, capsize=2, elinewidth=1,                
                color='lightcyan', ecolor='darkcyan', markeredgecolor="darkcyan",
                label='Sodankylä 67°N')

# DEM Processing
dem_co2_stats = norm_co2_DEM.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
dem_co2_se = dem_co2_stats['std'] / np.sqrt(dem_co2_stats['count'])
dem_co2_b = (abs(norm_co2_DEM.Diff)/norm_co2_DEM.CO2_gosat * 100).mean()
axs[0].errorbar(np.arange(len(dem_co2_stats)) + 1.0 - 0.1, dem_co2_stats['mean'],
                yerr=dem_co2_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                color='skyblue', ecolor='steelblue', markeredgecolor="steelblue",
                label='DEM 60°N')

# BRZ Processing
brz_co2_stats = norm_co2_brz.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
brz_co2_se = brz_co2_stats['std'] / np.sqrt(brz_co2_stats['count'])
axs[0].errorbar(np.arange(len(brz_co2_stats)) + 1.0 - 0.05, brz_co2_stats['mean'],
                color='c', ecolor='teal', markeredgecolor="teal",
                yerr=brz_co2_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                label='BRZ 56°N')

axs[0].errorbar(np.array(compare_co2_bi[2]), compare_co2_bi[0], 
                yerr=compare_co2_bi[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='moccasin', ecolor='orange', markeredgecolor="orange",
                label='Bremen 53°N')

axs[0].errorbar(np.array(compare_co2_br[2]) + 0.05, compare_co2_br[0], 
                yerr=compare_co2_br[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='khaki', ecolor='gold', markeredgecolor="gold",
                label='Bialystok 53°N')

axs[0].errorbar(np.array(compare_co2_pa[2]) + 0.1, compare_co2_pa[0], 
                yerr=compare_co2_pa[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='green', ecolor='darkgreen', markeredgecolor="darkgreen",      
                label='Park Falls 46°N')

# XCH4
axs[1].errorbar(np.array(compare_ch4_so[2]) - 0.15, compare_ch4_so[0], 
                yerr=compare_ch4_so[1], fmt='o', markersize=3, capsize=2, elinewidth=1,                
                color='lightcyan', ecolor='darkcyan', markeredgecolor="darkcyan",
                label='Sodankylä 67°N')

# DEM Processing
dem_ch4_stats = norm_ch4_DEM.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
dem_ch4_se = dem_ch4_stats['std'] / np.sqrt(dem_ch4_stats['count'])
axs[1].errorbar(np.arange(len(dem_ch4_stats)) + 1.0 - 0.1, dem_ch4_stats['mean'],
                yerr=dem_ch4_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                color='skyblue', ecolor='steelblue', markeredgecolor="steelblue",
                label='DEM 60°N')

# BRZ Processing
brz_ch4_stats = norm_ch4_BRZ.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
brz_ch4_se = brz_ch4_stats['std'] / np.sqrt(brz_ch4_stats['count'])
axs[1].errorbar(np.arange(len(brz_ch4_stats)) + 1.0 - 0.05, brz_ch4_stats['mean'],
                yerr=brz_ch4_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                                color='c', ecolor='teal', markeredgecolor="teal",
                label='BRZ 56°N')

axs[1].errorbar(np.array(compare_ch4_bi[2]), compare_ch4_bi[0], 
                yerr=compare_ch4_bi[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='moccasin', ecolor='orange', markeredgecolor="orange",
                label='Bremen 53°N')

axs[1].errorbar(np.array(compare_ch4_br[2]) + 0.05, compare_ch4_br[0], 
                yerr=compare_ch4_br[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='khaki', ecolor='gold', markeredgecolor="gold",
                label='Bialystok 53°N')

axs[1].errorbar(np.array(compare_ch4_pa[2]) + 0.1, compare_ch4_pa[0], 
                yerr=compare_ch4_pa[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='green', ecolor='darkgreen', markeredgecolor="darkgreen",
                label='Park Falls 46°N')

# Formatting
axs[1].set_xticks([1,2,3,4], ['MAM','JJA','SON','DNF'])
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].set_ylabel(r'$\Delta$XCO$_2$ [ppm]')
axs[1].set_ylabel(r'$\Delta$XCH$_4$ [ppb]')
axs[1].set_xlabel('Month')
axs[1].legend(ncol=2, loc='lower left', fontsize=6, handletextpad=0.05, columnspacing=0.1, markerscale=.5, borderaxespad=1)
plt.tight_layout()

# %% Figure 7, GOSAT vs. TCCON & JR-Station, box plot by site

fig, axs = plt.subplots(2,1, figsize=(18*ic_cm,15*ic_cm*.7), dpi=300, sharey='row', sharex='all')

# XCO2
axs[0].errorbar(np.array(compare_co2_so[2])/12, compare_co2_so[0], 
                yerr=compare_co2_so[1], fmt='o', markersize=3, capsize=2, elinewidth=1,                
                color='lightcyan', ecolor='darkcyan', markeredgecolor="darkcyan",
                label='Sodankylä 67°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.04, 0.96, 'SO 67°N',
            transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text((np.array(compare_co2_so[2])/12).min(), -1.8,
            f'b={compare_co2_so[4]:.2f}%', fontsize=5)

# DEM Processing
dem_co2_stats = norm_co2_dem.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
dem_co2_se = dem_co2_stats['std'] / np.sqrt(dem_co2_stats['count'])
dem_co2_b = (abs(norm_co2_dem.Diff)/norm_co2_dem.CO2_gosat *100).mean()
axs[0].errorbar((np.arange(len(dem_co2_stats))+1)/12 + .4, dem_co2_stats['mean'],
                yerr=dem_co2_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                color='skyblue', ecolor='steelblue', markeredgecolor="steelblue",
                label='DEM 60°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.175, 0.96, 'DEM 60°N',
              transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text(((np.arange(len(dem_co2_stats))+1)/12 + .4).min(), -1.8,
            f'b={dem_co2_b:.2f}%', fontsize=5)

# BRZ Processing
brz_co2_stats = norm_co2_brz.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
brz_co2_se = brz_co2_stats['std'] / np.sqrt(brz_co2_stats['count'])
brz_co2_b = (abs(norm_co2_brz.Diff)/norm_co2_brz.CO2_gosat *100).mean()
axs[0].errorbar((np.arange(len(brz_co2_stats))+1)/12 + .82, brz_co2_stats['mean'],
                color='c', ecolor='teal', markeredgecolor="teal",
                yerr=brz_co2_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                label='BRZ 56°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.335, 0.96, 'BRZ 56°N',
              transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text(((np.arange(len(brz_co2_stats))+1)/12 + .82).min(), -1.8,
            f'b={dem_co2_b:.2f}%', fontsize=5)

axs[0].errorbar(np.array(compare_co2_es[2])/12 + 1.15, compare_co2_es[0], 
                yerr=compare_co2_es[1], fmt='o', markersize=3, capsize=2, elinewidth=1,                
                color='yellowgreen', ecolor='olivedrab', markeredgecolor="olivedrab",
                label='East Trout Lake 54°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.465, 0.96, 'ET 54°N',
            transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text((np.array(compare_co2_es[2])/12).min() + 1.15, -1.8,
            f'b={compare_co2_es[4]:.2f}%', fontsize=5)

axs[0].errorbar(np.array(compare_co2_bi[2])/14 + 1.5, compare_co2_bi[0], 
                yerr=compare_co2_bi[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='moccasin', ecolor='orange', markeredgecolor="orange",
                label='Bialystok 53°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.6, 0.96, 'BI 53°N',
            transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text((np.array(compare_co2_bi[2])/14 + 1.55).min(), -1.8,
            f'b={compare_co2_bi[4]:.2f}%', fontsize=5)

axs[0].errorbar(np.array(compare_co2_br[2])/14 + 1.9, compare_co2_br[0], 
                yerr=compare_co2_br[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='khaki', ecolor='gold', markeredgecolor="gold",
                label='Bremen 53°N')
axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.75, 0.96, 'BR 53°N',
            transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text((np.array(compare_co2_br[2])/14 + 1.93).min(), -1.8,
            f'b={compare_co2_br[4]:.2f}%', fontsize=5)

axs[0].errorbar(np.array(compare_co2_pa[2])/14 + 2.26, compare_co2_pa[0], 
                yerr=compare_co2_pa[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='green', ecolor='darkgreen', markeredgecolor="darkgreen",      
                label='Park Falls 46°N')
# axs[0].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[0].text(0.88, 0.96, 'PA 46°N',
            transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
axs[0].text((np.array(compare_co2_pa[2])/14 + 2.28).min(), -1.8,
            f'b={compare_co2_pa[4]:.2f}%', fontsize=5)

# XCH4
axs[1].errorbar(np.array(compare_ch4_so[2])/12, compare_ch4_so[0], 
                yerr=compare_ch4_so[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='lightcyan', ecolor='darkcyan', markeredgecolor="darkcyan",
                label='Sodankylä 67°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text((np.array(compare_ch4_so[2])/12).min(), -16,
            f'b={compare_ch4_so[4]:.2f}%', fontsize=5)

# DEM Processing
dem_ch4_stats = norm_ch4_dem.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
dem_ch4_se = dem_ch4_stats['std'] / np.sqrt(dem_ch4_stats['count'])
dem_ch4_b = (abs(norm_ch4_dem.Diff)/norm_ch4_dem.CH4_gosat *100).mean()
axs[1].errorbar((np.arange(len(dem_ch4_stats))+1)/12 + .4, dem_ch4_stats['mean'],
                yerr=dem_ch4_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                color='skyblue', ecolor='steelblue', markeredgecolor="steelblue",
                label='DEM 60°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text(((np.arange(len(dem_ch4_stats))+1)/12 + .4).min(), -16,
            f'b={dem_ch4_b:.2f}%', fontsize=5)

# BRZ Processing
brz_ch4_stats = norm_ch4_brz.groupby('Season')['Diff'].agg(['count', 'mean', 'std'])
brz_ch4_se = brz_ch4_stats['std'] / np.sqrt(brz_ch4_stats['count'])
brz_ch4_b = (abs(norm_ch4_brz.Diff)/norm_ch4_brz.CH4_gosat *100).mean()
axs[1].errorbar((np.arange(len(brz_ch4_stats))+1)/12 + .82, brz_ch4_stats['mean'],
                yerr=brz_ch4_se, fmt='x', markersize=3, capsize=2, elinewidth=1,
                color='c', ecolor='teal', markeredgecolor="teal",
                label='BRZ 56°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text(((np.arange(len(brz_ch4_stats))+1)/12 + .82).min(), -16,
            f'b={brz_ch4_b:.2f}%', fontsize=5)

axs[1].errorbar(np.array(compare_ch4_es[2])/12 + 1.15, compare_ch4_es[0], 
                yerr=compare_ch4_es[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='yellowgreen', ecolor='olivedrab', markeredgecolor="olivedrab",
                label='East Trout Lake 54°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text((np.array(compare_ch4_es[2])/12).min() + 1.15, -16,
            f'b={compare_ch4_es[4]:.2f}%', fontsize=5)

axs[1].errorbar(np.array(compare_ch4_bi[2])/14 + 1.5, compare_ch4_bi[0], 
                yerr=compare_ch4_bi[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='moccasin', ecolor='orange', markeredgecolor="orange",
                label='Bialystok 53°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text((np.array(compare_ch4_bi[2])/14 + 1.55).min(), -16,
            f'b={compare_ch4_bi[4]:.2f}%', fontsize=5)

axs[1].errorbar(np.array(compare_ch4_br[2])/14 + 1.9, compare_ch4_br[0], 
                yerr=compare_ch4_br[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='khaki', ecolor='gold', markeredgecolor="gold",
                label='Bremen 53°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text((np.array(compare_ch4_br[2])/14 + 1.93).min(), -16,
            f'b={compare_ch4_br[4]:.2f}%', fontsize=5)

axs[1].errorbar(np.array(compare_ch4_pa[2])/14 + 2.26, compare_ch4_pa[0], 
                yerr=compare_ch4_pa[1], fmt='o', markersize=3, capsize=2, elinewidth=1,
                color='green', ecolor='darkgreen', markeredgecolor="darkgreen",
                label='Park Falls 46°N')
axs[1].axhline(y=0, color='black', linestyle='--', linewidth=.5)
axs[1].text((np.array(compare_ch4_pa[2])/14 + 2.28).min(), -16,
            f'b={compare_ch4_pa[4]:.2f}%', fontsize=5)

# Formatting
ticks_so = np.array(compare_co2_so[2])/12
ticks_dem = (np.arange(len(dem_ch4_stats))+1)/12 + .4
ticks_brz = (np.arange(len(brz_ch4_stats))+1)/12 + .82
ticks_es = np.array(compare_ch4_es[2])/12 + 1.15
ticks_br = np.array(compare_ch4_br[2])/14 + 1.9
ticks_bi = np.array(compare_ch4_bi[2])/14 + 1.5
ticks_pa = np.array(compare_ch4_pa[2])/14 + 2.26

ticks = np.concat((ticks_so, ticks_dem, ticks_brz, ticks_es, ticks_br, ticks_bi, ticks_pa))
labels = ['MAM', 'JJA', 'SON',
          'MAM', 'JJA', 'SON',
          'MAM', 'JJA', 'SON',
          'MAM', 'JJA', 'SON',
          'MAM', 'JJA', 'SON', 'DJF',
          'MAM', 'JJA', 'SON','DJF',
          'MAM', 'JJA', 'SON','DJF']

axs[1].set_xticks(ticks, labels)
axs[0].text(ticks_so[0]-.1, 2.5, "(a)", fontsize=10)
axs[1].text(ticks_so[0]-.1, 12, "(b)", fontsize=10)
axs[0].set_ylim(-2,3)
axs[1].set_ylim(-17,15)
axs[0].set_ylabel(r'$\Delta$XCO$_2$ [ppm]')
axs[1].set_ylabel(r'$\Delta$XCH$_4$ [ppb]')
plt.tight_layout()


# %% Figure 8. CAK vertical degradation and SE density distribution

fig, axs = plt.subplots(2,4, figsize=(18*ic_cm,15*ic_cm*.8),
                        gridspec_kw={'width_ratios': [0.8, 1, 1, 1]}, dpi=300, layout="constrained")

axs[1, 0].sharex(axs[0, 0])
axs[1, 0].sharey(axs[0, 0])

axs[0, 2].sharey(axs[0, 1])
axs[0, 3].sharey(axs[0, 1])
axs[0, 2].sharex(axs[0, 1])
axs[0, 3].sharex(axs[0, 1])
axs[0, 2].tick_params(labelleft=False)
axs[0, 3].tick_params(labelleft=False)

axs[1, 2].sharey(axs[1, 1])
axs[1, 3].sharey(axs[1, 1])
axs[1, 2].sharex(axs[1, 1])
axs[1, 3].sharex(axs[1, 1])
axs[1, 2].tick_params(labelleft=False)
axs[1, 3].tick_params(labelleft=False)
axs[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

seasons = ['MAM', 'JJA', 'SON']
sc = ['lightseagreen', 'firebrick', 'chocolate']

# CO2 at DEM
test = compare_gosat_co2_dem

max_counts = []
min_counts = []
x = np.concatenate(test.CAK.to_numpy())
y = np.concatenate(test.layer_se.to_numpy())

counts, _, _ = np.histogram2d(x, y, bins=(40, 40))
nonzero = counts[counts > 0]
if len(nonzero) > 0:
    max_counts.append(nonzero.max())
    min_counts.append(nonzero.min())

global_vmin = max(1, min(min_counts))
global_vmax = max(max_counts)
shared_norm = mcolors.LogNorm(vmin=global_vmin, vmax=global_vmax)

for s in [1,2,3]:
    x_err = []
    x_se = []
    y_mean = []
    
    x = test[test.Season==s].CAK
    x = np.concatenate(x.to_numpy())
    y = test[test.Season==s].Pressure_binned
    y = np.concatenate(y.to_numpy())
    
    for i in np.arange(0,len(x)/15):
        this_x_err = x[int(i)::15]
        x_err.append(this_x_err.mean())
        x_se.append(this_x_err.std()/len(this_x_err))
        
        y_mean.append(y[int(i)::15].mean())
    
    y_mean_idx = np.argsort(y_mean)
    y_mean = np.array(y_mean)[y_mean_idx]
    x_err = np.array(x_err)[y_mean_idx]
    
    # axs[0,0].errorbar(x_err, y_mean, xerr=x_se, fmt='^', markersize=2.5, 
    #                   elinewidth=.4, capsize=1)
    axs[0,0].plot(x_err, y_mean, 'o--', lw=.8, markersize=1.2, c=sc[s-1],label=seasons[s-1])
    
    y = test[test.Season==s].CAK
    y = np.concatenate(y.to_numpy())
    x = test[test.Season==s].layer_se
    x = np.concatenate(x.to_numpy())
    axs[0,s].hist2d(x, y, bins=(40, 40), cmap = "Greens",
                    norm = shared_norm)

# CH4 at DEM
test = compare_gosat_ch4_dem

counts, _, _ = np.histogram2d(x, y, bins=(40, 40))
nonzero = counts[counts > 0]
if len(nonzero) > 0:
    max_counts.append(nonzero.max())
    min_counts.append(nonzero.min())

global_vmin = max(1, min(min_counts))
global_vmax = max(max_counts)
shared_norm = mcolors.LogNorm(vmin=global_vmin, vmax=global_vmax)

for s in [1,2,3]:
    x_err = []
    x_se = []
    y_mean = []
    
    x = test[test.Season==s].CAK
    x = np.concatenate(x.to_numpy())
    y = test[test.Season==s].Pressure_binned
    y = np.concatenate(y.to_numpy())
    
    for i in np.arange(0,len(x)/15):
        this_x_err = x[int(i)::15]
        x_err.append(this_x_err.mean())
        x_se.append(this_x_err.std()/len(this_x_err))
        
        y_mean.append(y[int(i)::15].mean())
        
    y_mean_idx = np.argsort(y_mean)
    y_mean = np.array(y_mean)[y_mean_idx]
    x_err = np.array(x_err)[y_mean_idx]
    
    # axs[1,0].errorbar(x_err, y_mean, xerr=x_se, fmt='o', markersize=2.5, 
    #                   elinewidth=.4, capsize=1)
    axs[1,0].plot(x_err, y_mean, '^--', lw=.8, markersize=1.2, c=sc[s-1],label=seasons[s-1])
        
    y = test[test.Season==s].CAK
    y = np.concatenate(y.to_numpy())
    x = test[test.Season==s].layer_se
    x = np.concatenate(x.to_numpy())
    axs[1,s].hist2d(x, y, bins=(40, 40), cmap = "autumn_r",
                    norm = shared_norm)

axs[0,0].set_ylim(1050, 0)
axs[0,1].set_ylim(1.05, 0.3)
# axs[0,1].set_xlim(-0.1, 0.155)
axs[1,1].set_ylim(1.05, 0.72)
# axs[1,1].set_xlim(-2.2, 2)
axs[0,0].legend(ncol=1, fontsize=6, columnspacing=0.1, borderaxespad=1, loc=3)
axs[1,0].legend(ncol=1, fontsize=6, columnspacing=0.1, borderaxespad=1, loc=3)

cbar = fig.colorbar(axs[0, 3].collections[0], ax=axs[0, 1:4], orientation="vertical", pad=0.02)
cbar.set_label("Density")
cbar = fig.colorbar(axs[1, 3].collections[0], ax=axs[1, 1:4], orientation="vertical", pad=0.02)
cbar.set_label("Density")

axs[0,0].text(.88, 70, "(a)", fontsize=10)
axs[0,1].text(0.11, 0.35, "(b)", fontsize=10)
axs[0,2].text(0.11, 0.35, "(c)", fontsize=10)
axs[0,3].text(0.11, 0.35, "(d)", fontsize=10)
axs[0,1].text(-0.09, 0.35, "MAM", fontsize=10)
axs[0,2].text(-0.09, 0.35, "JJA", fontsize=10)
axs[0,3].text(-0.09, 0.35, "SON", fontsize=10)

axs[1,0].text(.88, 70, "(e)", fontsize=10)
axs[1,1].text(1.22, 0.745, "(f)", fontsize=10)
axs[1,2].text(1.22, 0.745, "(g)", fontsize=10)
axs[1,3].text(1.22, 0.745, "(h)", fontsize=10)

axs[0,0].set_xlabel(r"$\mathrm{CAK}_{\mathrm{CO2}}$")
axs[1,0].set_xlabel(r"$\mathrm{CAK}_{\mathrm{CH4}}$")
axs[0,2].set_xlabel(r"$\mathrm{SE}_{\mathrm{CO2}}$")
axs[1,2].set_xlabel(r"$\mathrm{SE}_{\mathrm{CH4}}$")
axs[0,1].set_ylabel(r"$\mathrm{CAK}_{\mathrm{CO2}}$")
axs[1,1].set_ylabel(r"$\mathrm{CAK}_{\mathrm{CH4}}$")
axs[0,0].set_ylabel("Pressure [hPa]")
axs[1,0].set_ylabel("Pressure [hPa]")

# %% Figure 9. Total SE vs. SNR ratio

fig, axs = plt.subplots(2,3, figsize=(18*ic_cm,15*ic_cm*.8),
                        dpi=300, sharex='col', layout='constrained')
axs[1, 0].sharey(axs[0, 0])
axs[0, 1].tick_params(labelleft=False)
axs[1, 1].sharey(axs[1, 0])
axs[1, 1].tick_params(labelleft=False)
labels = ['MAM','JJA','SON']


test = compare_gosat_ch4_dem

for s in [1,2,3]:
    test_ls = test[test.Season==s].total_se.values
    test_snr = test[test.Season==s].snrch4.values
    test_sza = test[test.Season==s].sza.values
    test_diff = test[test.Season==s].Diff.values
    test_aot = test[test.Season==s].aot.values
    
    axs[0,0].scatter(test_sza, test_snr, s=5, c=sc[s-1],alpha=.4, label=labels[s-1])
    axs[0,1].scatter(test_aot, test_snr, s=5, c=sc[s-1],alpha=.4, label=labels[s-1])

test = pd.merge(compare_gosat_ch4_dem, long_ch4_res_dem[['Date','GHG_merra2']])
for s in [1,2,3]:
    test_diff = test[test.Season==s].CH4_gosat.values - test[test.Season==s].GHG_merra2.values
    test_snr = test[test.Season==s].snrch4.values
    test_ls = test[test.Season==s].total_se.values
    
    axs[0,2].scatter(test_ls, test_diff, s=5, c=test_snr, cmap="viridis", alpha=.4)

# BRZ
test = compare_gosat_ch4_brz

for s in [1,2,3]:
    test_ls = test[test.Season==s].total_se.values
    test_snr = test[test.Season==s].snrch4.values
    test_sza = test[test.Season==s].sza.values
    test_diff = test[test.Season==s].Diff.values
    test_aot = test[test.Season==s].aot.values
    
    axs[1,0].scatter(test_sza, test_snr, s=5, c=sc[s-1], alpha=.4, label=labels[s-1])
    axs[1,1].scatter(test_aot, test_snr, s=5, c=sc[s-1],alpha=.4, label=labels[s-1])
    # axs[1,2].scatter(test_snr, test_ls, s=5, c=test_diff, cmap="viridis", alpha=.4)

for s in [1,2,3]:
    test_diff = test[test.Season==s].Diff.values
    test_snr = test[test.Season==s].snrch4.values
    test_ls = test[test.Season==s].total_se.values
    ana.draw_confidence_ellipse(test_ls, test_diff, axs[1,2], n_std=2.0, facecolor='none')
    axs[1,2].scatter(test_ls, test_diff, s=5, c=test_snr, cmap="viridis", alpha=.4)

axs[1,0].set_xlabel('SZA [°]')
axs[0,0].set_ylabel('SNR [%]')
axs[1,0].set_ylabel('SNR [%]')
axs[1,1].set_xlabel('AOT')
axs[0,2].set_ylabel(rf'$\Delta$XCH$_4$ [ppb]')
axs[1,2].set_ylabel(rf'$\Delta$XCH$_4$ [ppb]')
axs[1,2].set_xlabel('Total SE [ppb]')
# axs[0,0].set_ylim(1.1, .25)
# axs[0,0].text(-.05, .35, "DEM", fontsize=10)
# axs[1,0].text(-.05, .35, "BRZ", fontsize=10)


# %% DOFS vs. SA nir

fig, axs = plt.subplots(2,3, figsize=(18*ic_cm,15*ic_cm*.8),
                        dpi=300, sharey='all', sharex='all')
labels = ['MAM','JJA','SON']

test = compare_gosat_ch4_dem

for s in [1,2,3]:
    test_ls = test[test.Season==s].layer_se.values
    test_cak = test[test.Season==s].CAK.values
    for i in range(len(test_ls)):
        axs[0,s-1].scatter(test_ls[i], test_cak[i], s=2, c='grey', alpha=.4)

test = compare_gosat_ch4_brz

for s in [1,2,3]:
    test_ls = test[test.Season==s].layer_se.values
    test_cak = test[test.Season==s].CAK.values
    for i in range(len(test_ls)):
        axs[1,s-1].scatter(test_ls[i], test_cak[i], s=2, c='grey', alpha=.4)

axs[0,0].set_ylabel('SA$_NIR$')
axs[1,0].set_ylabel('SA$_NIR$')
axs[1,0].set_xlabel('SE [ppm]')
axs[1,1].set_xlabel('SE [ppm]')
axs[1,2].set_xlabel('SE [ppm]')
axs[0,0].set_ylim(1.1, .25)
axs[0,0].text(-.05, .35, "DEM", fontsize=10)
axs[1,0].text(-.05, .35, "BRZ", fontsize=10)
axs[0,0].set_title('MAM', fontsize=10)
axs[0,1].set_title('JJA', fontsize=10)
axs[0,2].set_title('SON', fontsize=10)

# %%

fig, axs = plt.subplots(3, 3, figsize=(18*ic_cm,15*ic_cm), dpi=300, sharey='all', sharex='all')

test = compare_gosat_ch4_dem

for s in [1,2,3]:
    test_ls = test[test.Season==s].Diff.values
    test_cak = test[test.Season==s].xh2o.values
    axs[1,s-1].scatter(test_ls, test_cak)

test = compare_gosat_ch4_brz

for s in [1,2,3]:
    test_ls = test[test.Season==s].Diff.values
    test_cak = test[test.Season==s].xh2o.values
    axs[2,s-1].scatter(test_ls, test_cak)
    
# %%
fig, axs = plt.subplots(3, 4, figsize=(9,7), sharey = 'all', sharex = 'all', dpi=300)
colors = ['seagreen', 'orangered', 'khaki', 'steelblue']

# years = column_tp_adj2_DEM.Year.unique()
years = long_ch4_str_adj_brz.Year.unique()
x = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
y = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

for year in range(len(years)):
    # test = column_tp_adj2_DEM.loc[column_tp_adj2_DEM.Year == years[year]]    
    test = long_ch4_str_adj_brz.loc[long_ch4_str_adj_brz.Year == years[year]]
    for i in range(len(test)):
        axs[x[year],y[year]].plot(test.iloc[i].binned_ghg, test.iloc[i].bins,
                                      c='grey', alpha=.3, linewidth = .7)

axs[0,0].set_ylim(1050, 100) 
axs[0,0].set_xlim(1800, 2200)

plt.tight_layout()