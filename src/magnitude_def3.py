def compute_evolution_ratio(log2fc: pd.DataFrame, 
                            logFC_cutoff: float,
                            tolerance: float) -> tuple: #in this case 0.5
    arr = log2fc.values
    dirs = (arr > logFC_cutoff).astype(int) - (arr < -logFC_cutoff).astype(int) # filter for genes that satisfy first 2 definitions of repeatability
                              
    same_dir = (dirs.max(axis=1) == dirs.min(axis=1)) & (dirs.max(axis=1) != 0) 
    similar_mag = (arr.max(axis=1) - arr.min(axis=1)) < tolerance
    evolved = same_dir & similar_mag
    num_evolved = evolved.sum()

    changed_per_strain = (dirs != 0).sum(axis=0)
    avg_changed = changed_per_strain.mean()

    return num_evolved, avg_changed

def simulate_null_distribution(log2fc: pd.DataFrame,
                               logFC_cutoff: float,
                               tolerance: float,
                               num_simulations: int = 1000) -> np.ndarray:
    nulls = np.zeros(num_simulations)
    arr = log2fc.values
    n_genes, n_strains = arr.shape

    for i in range(num_simulations):
        # Column-wise permutation
        shuffled = np.column_stack([np.random.permutation(arr[:, j]) for j in range(n_strains)])
        df_shuf = pd.DataFrame(shuffled, index=log2fc.index, columns=log2fc.columns)
        num_evolved, _ = compute_evolution_ratio(df_shuf, logFC_cutoff, tolerance)
        nulls[i] = num_evolved

    return nulls

def plot_simulation_results(simulated_counts, observed_count, title, ax):
    mean_sim = simulated_counts.mean()
    std_sim = simulated_counts.std(ddof=0)
    D = (observed_count - mean_sim) / std_sim if std_sim > 0 else np.nan

    ax.hist(simulated_counts, bins=30, alpha=0.75)
    ax.axvline(x=observed_count, color='red', 
               label=f'Observed: {observed_count:.0f}')
    ax.set_title(title)
    ax.set_xlabel("Number of genes evolved in the same direction")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.text(0.95, 0.85, f"D = {D:.2f} SD", transform=ax.transAxes,
            va='top', ha='right', fontsize=9)
