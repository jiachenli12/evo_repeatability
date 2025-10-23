import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _summarize_df(df):
    n_series = df['Strain Combination'].astype(str).str.split(',').str.len()
    obs = df['Observed Ratio (%)'].to_numpy() / 100.0
    sim = df['Simulated Mean (%)'].to_numpy() / 100.0
    grp = pd.DataFrame({'n': n_series, 'obs': obs, 'sim': sim})
    stats = grp.groupby('n', sort=True).agg(
        obs_mean=('obs', 'mean'),
        sim_mean=('sim', 'mean'),
        obs_se=('obs', lambda x: np.std(x, ddof=1)/np.sqrt(len(x))),
        sim_se=('sim', lambda x: np.std(x, ddof=1)/np.sqrt(len(x)))
    ).reset_index()
    Ns = stats['n'].to_numpy()
    return Ns, stats['obs_mean'].to_numpy(), stats['sim_mean'].to_numpy(), stats['obs_se'].to_numpy(), stats['sim_se'].to_numpy()

def _split_sig(Ds, threshold=1.96):
    Ds = np.asarray(Ds, dtype=float)
    return np.abs(Ds) < threshold

def _scatter(ax, x_center, Ds, color, jitter=0.02):
    Ds = np.asarray(Ds, dtype=float)
    if Ds.size == 0: return
    nonsig = _split_sig(Ds)
    x = x_center + np.random.normal(scale=jitter, size=len(Ds))
    for i, (xi, val) in enumerate(zip(x, Ds)):
        if nonsig[i]:
            ax.scatter(xi, val, s=750, marker='o', edgecolor='grey', facecolor='grey', alpha=0.75, zorder=2)
        else:
            ax.scatter(xi, val, s=750, marker='o', color=color, edgecolor=color, alpha=0.75, zorder=2)

def _draw_one(ax_bar, ax_pts, df, group_Ds, color, title):
    group_spacing = 2.0
    bar_width = 0.6
    dot_offset = 0.7
    dot_jitter = 0.02
    err_elinewidth = 3
    err_capthick = 3
    capsize = 10
    fontsize_axis = 50
    fontsize_title = 56
    Ns, obs_m, sim_m, obs_se, sim_se = _summarize_df(df)
    x = np.arange(len(Ns)) * group_spacing
    width = bar_width
    err_kwargs = dict(elinewidth=err_elinewidth, capthick=err_capthick, capsize=capsize)
    ax_bar.bar(x - width/2, obs_m, width, yerr=obs_se, error_kw=err_kwargs, color=color)
    ax_bar.bar(x + width/2, sim_m, width, yerr=sim_se, error_kw=err_kwargs, color='grey')
    ax_bar.set_ylabel("Mean Dice’s coefficient", fontsize=fontsize_axis)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{n} lines" for n in Ns], fontsize=fontsize_axis)
    ax_bar.tick_params(axis='y', labelsize=fontsize_axis, length=15, width=3)
    ax_bar.tick_params(axis='x', labelsize=fontsize_axis, length=15, width=3)
    for i, n in enumerate(Ns):
        Ds = group_Ds.get(n, [])
        if Ds: _scatter(ax_pts, x[i] + width/2 + dot_offset, Ds, color, jitter=dot_jitter)
    ax_pts.set_ylabel(r"$\it{Z}$-score", fontsize=fontsize_axis)
    ax_pts.tick_params(axis='y', labelsize=fontsize_axis, length=15, width=3)
    ax_pts.set_ylim(-1, None)
    sns.despine(ax=ax_bar, top=True, right=False, left=False, bottom=False)
    sns.despine(ax=ax_pts, top=True, right=False, left=False, bottom=False)
    ax_bar.set_title(title, fontsize=fontsize_title, pad=30)

def plot_six(df_A_deg, df_A_dir, df_A_mag,
             df_B_deg, df_B_dir, df_B_mag,
             group_Ds_A_deg, group_Ds_A_dir, group_Ds_A_mag,
             group_Ds_B_deg, group_Ds_B_dir, group_Ds_B_mag):
    colors = ['dodgerblue', 'orange', 'red']
    titles_A = [r"$\it{E.\ coli}$ Founder A",
                r"$\it{E.\ coli}$ Founder A",
                r"$\it{E.\ coli}$ Founder A"]
    titles_B = [r"$\it{E.\ coli}$ Founder B",
                r"$\it{E.\ coli}$ Founder B",
                r"$\it{E.\ coli}$ Founder B"]
    dfs_A = [df_A_deg, df_A_dir, df_A_mag]
    dfs_B = [df_B_deg, df_B_dir, df_B_mag]
    Ds_A = [group_Ds_A_deg, group_Ds_A_dir, group_Ds_A_mag]
    Ds_B = [group_Ds_B_deg, group_Ds_B_dir, group_Ds_B_mag]
    fig, axes = plt.subplots(2, 3, figsize=(60, 26))
    for j in range(3):
        axA = axes[0, j]
        axB = axes[1, j]
        _draw_one(axA, axA.twinx(), dfs_A[j], Ds_A[j], colors[j], titles_A[j])
        _draw_one(axB, axB.twinx(), dfs_B[j], Ds_B[j], colors[j], titles_B[j])
    plt.subplots_adjust(wspace=0.35, hspace=0.45, left=0.05, right=0.99, top=0.98, bottom=0.06)
    plt.show()

group_Ds_A_deg = {2: [10.35, 25.6, 3.81], 3: [13.1]}
group_Ds_B_deg = {2: [14.11, 1.47, 4.28], 3: [7.39]}
group_Ds_A_dir = {2: [14.36, 6.65, 36.41], 3: [28.64]}
group_Ds_B_dir = {2: [1.94, 3.91, 19.64], 3: [6.67]}
group_Ds_A_mag = {2: [14.21, 5.98, 33.38], 3: [15.91]}
group_Ds_B_mag = {2: [0.51, 4.03, 19.77], 3: [-0.15]}

df_A_deg = pd.DataFrame({'Strain Combination': ['A1,A2','A1,A3','A2,A3','A1,A2,A3'],
                         'Observed Ratio (%)':[15.6,7.8,40.4,3.2],
                         'Simulated Mean (%)':[2.8,2.4,4.0,0.09]})
df_A_dir = pd.DataFrame({'Strain Combination': ['A1,A2','A1,A3','A2,A3','A1,A2,A3'],
                         'Observed Ratio (%)':[14.4,7.8,39.4,3.2],
                         'Simulated Mean (%)':[1.33,1.23,2.02,0.03]})
df_A_mag = pd.DataFrame({'Strain Combination': ['A1,A2','A1,A3','A2,A3','A1,A2,A3'],
                         'Observed Ratio (%)':[13.75,6.90,34.72,1.92],
                         'Simulated Mean (%)':[1.25,1.13,1.94,0.02]})
df_B_deg = pd.DataFrame({'Strain Combination': ['B1,B2','B1,B3','B2,B3','B1,B2,B3'],
                         'Observed Ratio (%)':[4.9,8.2,22.8,1.7],
                         'Simulated Mean (%)':[2.6,2.1,2.8,0.06]})
df_B_dir = pd.DataFrame({'Strain Combination': ['B1,B2','B1,B3','B2,B3','B1,B2,B3'],
                         'Observed Ratio (%)':[3.25,5.13,21.18,0.86],
                         'Simulated Mean (%)':[1.34,1.09,1.31,0.02]})
df_B_mag = pd.DataFrame({'Strain Combination': ['B1,B2','B1,B3','B2,B3','B1,B2,B3'],
                         'Observed Ratio (%)':[1.63,5.13,20.39,0.00],
                         'Simulated Mean (%)':[1.15,1.01,1.19,0.02]})

plot_six(
    df_A_deg, df_A_dir, df_A_mag,
    df_B_deg, df_B_dir, df_B_mag,
    group_Ds_A_deg, group_Ds_A_dir, group_Ds_A_mag,
    group_Ds_B_deg, group_Ds_B_dir, group_Ds_B_mag
)
