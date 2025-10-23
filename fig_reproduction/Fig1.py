#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

gly_2lines_d_deg = [57.51, 34.36, 29.14, 32.74, 30.63, 63.63, 42.06, 36.07, 38.0, 35.91, 57.61, 56.43, 62.22, 60.14, 36.2, 63.02, 62.37, 31.32, 63.72, 33.28, 31.28]
gly_3lines_d_deg = [77.04, 74.45, 72.7, 72.35, 114.54, 51.87, 53.69, 49.91, 85.75, 46.78, 48.51, 80.9, 47.9, 82.48, 85.54, 60.06, 60.88, 60.05, 75.49, 56.66, 55.82, 70.29, 55.04, 74.64, 73.97, 83.33, 80.52, 51.74, 83.33, 54.91, 51.59, 86.48, 48.95, 48.68, 48.43]
gly_4lines_d_deg = [93.1, 102.27, 93.63, 147.48, 91.33, 95.15, 139.39, 92.7, 145.64, 135.82, 68.21, 68.21, 105.81, 70.64, 110.54, 111.55, 65.78, 107.53, 105.24, 107.38, 80.56, 78.06, 96.24, 79.92, 100.9, 99.26, 77.21, 94.35, 94.84, 93.02, 106.97, 74.65, 67.22, 69.58, 68.31]
gly_5lines_d_deg = [119.97, 119.05, 192.2, 129.43, 185.13, 186.65, 119.15, 181.46, 178.66, 182.61, 88.5, 143.01, 138.47, 141.5, 123.63, 105.96, 128.23, 119.92, 124.21, 120.89, 91.86]
gly_6lines_d_deg = [159.27, 241.59, 241.34, 237.13, 234.44, 164.77, 160.68]
gly_7lines_d_deg = [296.89]
lac_2lines_d_deg = [31.88, 42.16, 35.79, 36.5, 37.64, 50.65, 52.34, 61.83, 62.27, 59.03, 43.8, 56.08, 58.81, 59.8, 52.12, 66.0, 64.01, 45.44, 66.98, 47.53, 47.1]
lac_3lines_d_deg = [108.35, 91.95, 89.73, 93.93, 132.76, 113.8, 111.54, 121.66, 163.22, 105.52, 106.58, 138.92, 104.25, 137.79, 141.97, 148.09, 149.23, 159.46, 143.53, 178.48, 167.06, 124.41, 165.9, 126.45, 122.83, 155.72, 164.16, 141.87, 172.12, 145.47, 149.61, 180.73, 131.12, 135.4, 140.27]
lac_4lines_d_deg = [295.8, 293.97, 305.27, 404.16, 258.75, 264.76, 379.56, 264.52, 367.13, 359.22, 308.37, 310.23, 440.05, 329.01, 450.27, 459.24, 285.7, 374.91, 378.53, 388.78, 414.73, 407.38, 371.18, 437.92, 375.37, 381.92, 472.02, 342.63, 343.01, 365.04, 456.31, 393.24, 409.67, 410.75, 357.55]
lac_5lines_d_deg = [793.85, 713.54, 1108.25, 878.63, 1215.14, 1073.0, 767.88, 1011.89, 1108.25, 1009.87, 837.5, 1240.65, 1125.07, 1438.71, 985.17, 1170.33, 1034.34, 1194.03, 992.02, 1008.92, 1039.17]
lac_6lines_d_deg = [2087.82, 2408.41, 2946.75, 3400.93, 2636.97, 5884.75, 2633.46]
lac_7lines_d_deg = [4163.21]
gly_lac_d_deg = [23.10, 47.76, 37.52, 40.67, 43.80, 42.13, 29.72, 18.96, 38.93, 32.12, 35.43, 35.94, 35.58, 25.63, 12.40, 24.68, 19.82, 21.77, 23.70, 21.54, 15.41, 10.69, 20.11, 17.42, 20.21, 19.55, 19.26, 14.07, 10.88, 22.96, 18.23, 21.06, 20.03, 20.73, 14.96, 10.71, 21.81, 17.29, 19.44, 19.80, 19.96, 13.46, 22.16, 43.90, 36.37, 42.98, 41.97, 39.12, 28.36]

gly_d_dir = [35.96, 14.42, 15.97, 14.19, 11.12, 37.08, 20.60, 17.49, 19.79, 13.20, 30.73, 22.25,26.76, 24.29, 19.32, 27.35, 26.20, 14.00, 30.76, 18.85, 17.42]
lac_d_dir = [24.63, 27.46, 22.42, 28.06, 25.69, 35.27, 35.22, 39.88, 38.49, 40.06, 28.24, 41.23, 40.99, 41.44, 36.83, 43.09, 41.93, 29.55, 42.55, 32.02, 34.04]
gly_lac_d_dir = [13.78, 27.60, 24.09, 28.88, 25.63, 23.33, 16.46, 13.30, 22.97, 19.42, 23.53, 24.98, 18.83, 16.08, 7.65, 11.12, 12.68, 9.30, 12.70, 13.64, 11.60, 5.88, 12.53, 9.85, 9.35, 11.94, 13.08, 8.32, 7.34, 12.11, 12.42, 6.22, 11.96, 16.41, 11.18, 7.68, 11.22, 11.42, 7.47, 13.56, 13.40, 11.04, 15.51, 25.84, 23.23, 22.37, 27.18, 27.91, 20.32]

gly_2lines_d_mag = [26.34, 16.07, 16.94, 14.70, 11.73, 35.78, 23.82, 21.33, 24.61, 21.67, 24.57, 28.64, 31.52, 31.13, 18.52, 31.79, 31.91, 13.99, 35.45, 18.04, 16.99]
gly_3lines_d_mag = [29.96, 28.62, 31.36, 26.11, 41.38, 19.57, 19.17, 16.24, 36.31, 19.78, 16.17, 33.64, 16.68, 34.88, 31.59, 26.81, 31.16, 28.16, 31.77, 28.98, 26.06, 24.05, 29.94, 29.28, 26.75, 38.29, 36.25, 19.96, 42.31, 23.02, 21.61, 42.05, 19.43, 18.03, 22.40]
gly_4lines_d_mag = [29.86, 36.57, 30.51, 47.93, 34.64, 26.11, 40.24, 32.80, 43.78, 38.36, 21.10, 16.99, 31.40, 17.09, 36.85, 29.78, 20.44, 38.04, 32.86, 34.31, 32.55, 29.26, 29.25, 33.21, 34.23, 29.98, 33.14, 28.17, 29.18, 33.37, 46.42, 22.77, 18.37, 22.74, 22.20]
gly_5lines_d_mag = [36.15, 26.70, 42.05, 32.29, 60.49, 42.37, 31.88, 52.92, 44.54, 48.43, 18.56, 36.70, 25.11, 32.25, 39.55, 35.74, 31.24, 30.27, 34.80, 34.07, 20.14]
gly_6lines_d_mag = [33.28, 58.82, 42.15, 53.34, 55.72, 29.75, 34.72]
gly_7lines_d_mag = [51.66]
lac_2lines_d_mag = [19.19, 22.48, 17.87, 23.68, 20.63, 26.90, 29.34, 36.51, 35.12, 35.37, 20.52, 32.92, 32.23, 35.30, 34.05, 38.13, 39.89, 22.99, 40.75, 25.05, 29.60]
lac_3lines_d_mag = [34.21, 24.90, 38.73, 38.57, 41.39, 21.91, 51.06, 51.18, 60.77, 37.98, 38.85, 39.39, 46.07, 55.53, 50.30, 51.01, 48.99, 55.02, 43.74, 66.96, 77.68, 43.14, 67.80, 35.37, 47.18, 60.83, 66.54, 52.06, 67.68, 58.00, 85.90, 79.38, 39.04, 46.59, 57.22]
lac_4lines_d_mag = [42.26, 70.14, 105.79, 64.67, 44.03, 103.51, 70.82, 86.18, 79.15, 83.88, 56.64, 44.81, 51.72, 120.50, 127.53, 107.87, 127.94, 90.56, 71.88, 79.34, 79.89, 95.00, 100.40, 84.25, 66.56, 125.39, 136.04, 75.08, 93.24, 96.41, 138.08, 74.84, 142.77, 178.84, 81.89]
lac_5lines_d_mag = [63.25, 111.87, 189.80, 67.10, 156.64]
glylac_d_mag = [3.07, 3.09, 5.18, 10.78, 6.11, 10.48, 13.45, 4.99, 0.79, 4.74, 7.13, 5.65, 7.52, 11.79, 2.70, 2.02, 4.86, 7.05, 4.44, 9.64, 10.30, 2.71, 1.61, 3.48, 2.85, 5.12, 7.41, 8.49, 10.08, 13.56, 17.53, 20.60, 21.00, 22.20, 24.92, 2.73, -0.87, 0.34, 1.82, 1.98, 5.05, 8.03, 11.59, 11.40, 18.40, 24.67, 21.26, 21.09, 25.79]

group_Ds_same_genes = {"gly":{2:gly_2lines_d_deg,3:gly_3lines_d_deg,4:gly_4lines_d_deg,5:gly_5lines_d_deg,6:gly_6lines_d_deg,7:gly_7lines_d_deg},
                       "lac":{2:lac_2lines_d_deg,3:lac_3lines_d_deg,4:lac_4lines_d_deg,5:lac_5lines_d_deg,6:lac_6lines_d_deg,7:lac_7lines_d_deg},
                       "gly_lac":{2:gly_lac_d_deg}}
group_Ds_same_direction = {"gly":{2:gly_d_dir,3:ecoli_gly_3_strains_d_pval,4:ecoli_gly_4_strains_d_pval,5:ecoli_gly_5_strains_d_pval,6:ecoli_gly_6_strains_d_pval,7:ecoli_gly_7_strains_d_pval},
                           "lac":{2:lac_d_dir,3:lac_3lines_d_deg,4:lac_4lines_d_deg,5:lac_5lines_d_deg,6:lac_6lines_d_deg,7:lac_7lines_d_deg},
                           "gly_lac":{2:gly_lac_d_dir}}
group_Ds_same_magnitude = {"gly":{2:gly_2lines_d_mag,3:gly_3lines_d_mag,4:gly_4lines_d_mag,5:gly_5lines_d_mag,6:gly_6lines_d_mag,7:gly_7lines_d_mag},
                           "lac":{2:lac_2lines_d_mag,3:lac_3lines_d_mag,4:lac_4lines_d_mag,5:lac_5lines_d_mag},
                           "gly_lac":{2:glylac_d_mag}}

data_dir = Path("/evo_repeatability/fig_reproduction/Fig1_data")

def _summarize(path):
    df = pd.read_csv(path, sep="\t")
    go, gs = {}, {}
    for _, row in df.iterrows():
        n = len(str(row["Strain Combination"]).split(","))
        go.setdefault(n, []).append(row["Observed Ratio (%)"] / 100)
        gs.setdefault(n, []).append(row["Simulated Mean (%)"] / 100)
    Ns = sorted(go)
    obs_means = [np.mean(go[n]) for n in Ns]
    sim_means = [np.mean(gs[n]) for n in Ns]
    obs_ses = [np.std(go[n], ddof=1) / np.sqrt(len(go[n])) for n in Ns]
    sim_ses = [np.std(gs[n], ddof=1) / np.sqrt(len(gs[n])) for n in Ns]
    return Ns, obs_means, sim_means, obs_ses, sim_ses, go


def _panel(ax1, output_file, group_Ds, bar_color, point_color,
           group_spacing=2.0, bar_width=0.6, dot_offset=0.7, dot_jitter=0.02,
           error_elinewidth=3, error_capthick=3, tick_font=24, dot_size=110):
    Ns, obs_means, sim_means, obs_ses, sim_ses, go = _summarize(output_file)

    base_x = np.arange(len(Ns), dtype=float)
    x = base_x * group_spacing
    w = bar_width

    err_kwargs = dict(elinewidth=error_elinewidth, capthick=error_capthick, capsize=10)
    ax1.bar(x - w/2, obs_means, w, yerr=obs_ses, error_kw=err_kwargs, color=bar_color)
    ax1.bar(x + w/2, sim_means, w, yerr=sim_ses, error_kw=err_kwargs, color="grey")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} lines" for n in Ns], fontsize=tick_font, rotation=30)
    ax1.tick_params(axis="y", labelsize=tick_font, length=10, width=2.5)
    ax1.tick_params(axis="x", labelsize=tick_font, length=10, width=2.5)

    ax2 = ax1.twinx()
    for i, n in enumerate(Ns):
        Ds = np.array(group_Ds.get(n, []), dtype=float)
        if Ds.size == 0:
            continue
        xpos_base = x[i] + w/2 + dot_offset
        xpos = xpos_base + np.random.normal(scale=dot_jitter, size=Ds.size)
        colors = np.where(Ds < 1.96, "grey", point_color)
        ax2.scatter(xpos, Ds, c=colors, alpha=0.9, s=dot_size, edgecolors=colors, zorder=15)

    ax2.set_ylim(-2, None)
    ax2.set_ylabel(r"$\it{Z}$-score", fontsize=tick_font)
    ax2.tick_params(axis="y", labelsize=tick_font, length=10, width=2.5)
    ax2.tick_params(axis="x", labelsize=tick_font, length=10, width=2.5)

    sns.despine(ax=ax1, top=True, right=False, left=False, bottom=False)
    sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False)
    return Ns, x, w, obs_means, obs_ses, go


data_dir = Path("/Users/jiachenli/Desktop/Fig1_data")

rows = [
    ("same_genes", "dodgerblue",
     {"gly": "gly_same_genes.txt", "lac": "lac_same_genes.txt", "gly_lac": "gly_lac_same_genes.txt"},
     group_Ds_same_genes),
    ("same_direction", "orange",
     {"gly": "gly_same_direction.txt", "lac": "lac_same_direction.txt", "gly_lac": "gly_lac_same_direction.txt"},
     group_Ds_same_direction),
    ("same_magnitude", "red",
     {"gly": "gly_same_magnitude.txt", "lac": "lac_same_magnitude.txt", "gly_lac": "gly_lac_same_magnitude.txt"},
     group_Ds_same_magnitude),
]

col_titles = [
    ("gly",     r"$\it{E.\ coli}$ in glycerol medium"),
    ("lac",     r"$\it{E.\ coli}$ in lactate medium"),
    ("gly_lac", None),  # no grand title for between environments
]

fig, axes = plt.subplots(3, 3, figsize=(30, 22), gridspec_kw={"wspace": 0.28, "hspace": 0.65})

for r, (metric_key, color, file_map, z_map) in enumerate(rows):
    for c, (col_key, col_title) in enumerate(col_titles):
        ax1 = axes[r, c]
        fpath = data_dir / file_map[col_key]
        zdict = z_map[col_key]

        Ns, x, w, obs_means, obs_ses, _ = _panel(ax1, str(fpath), zdict, color, color, tick_font=24, dot_size=110)

        if col_title:
            ax1.set_title(col_title, fontsize=28, pad=10)
        if c == 0:
            ax1.set_ylabel("Mean Dice’s coefficient", fontsize=24)
        else:
            ax1.set_ylabel("")

        # Right-column: add black "Gly vs. lac" and reference lines
        if c == 2 and len(obs_means) > 0:
            y_top = max(np.array(obs_means) + np.array([s if not np.isnan(s) else 0 for s in obs_ses])) + 0.02
            x_shift = -0.25 
            ax1.text(np.mean(x) + x_shift, y_top, "gly vs. lac",
            ha="center", va="bottom", fontsize=22,
            color="black", zorder=20)

            gly_path = data_dir / file_map["gly"]
            lac_path = data_dir / file_map["lac"]
            _, _, _, _, _, go_gly = _summarize(gly_path)
            _, _, _, _, _, go_lac = _summarize(lac_path)
            gly_mean_2 = np.mean(go_gly.get(2, [np.nan]))
            lac_mean_2 = np.mean(go_lac.get(2, [np.nan]))

            pad = 0.012
            x_left, x_right = ax1.get_xlim()
            x_mid = 0.5 * (x_left + x_right)

            if not np.isnan(gly_mean_2):
                ax1.axhline(gly_mean_2, color=color, linewidth=3)
            if not np.isnan(lac_mean_2):
                ax1.axhline(lac_mean_2, color=color, linewidth=3)

            if not np.isnan(gly_mean_2) and not np.isnan(lac_mean_2):
                hi, lo = max(gly_mean_2, lac_mean_2), min(gly_mean_2, lac_mean_2)
                if metric_key in ("same_genes", "same_direction"):
                    ax1.text(x_mid, hi + pad, "gly vs. gly", ha="center", va="bottom", fontsize=22, color="black")
                    ax1.text(x_mid, lo - pad, "lac vs. lac", ha="center", va="top", fontsize=22, color="black")
                else:
                    ax1.text(x_mid, gly_mean_2 + pad, "Gly vs. gly", ha="center", va="bottom", fontsize=22, color="black")
                    ax1.text(x_mid, lac_mean_2 + pad, "Lac vs. lac", ha="center", va="bottom", fontsize=22, color="black")

plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, wspace=0.28, hspace=0.7)
plt.show()



