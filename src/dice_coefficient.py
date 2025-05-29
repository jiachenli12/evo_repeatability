def compute_dice(direction_subset):
    """Compute (genes changed in any direction) / (average changed genes across strains)."""
    num_genes_changed_all = count_any_direction(direction_subset)

    changed_genes_per_strain = np.sum(direction_subset != 0, axis=0)
    avg_changed_genes = np.mean(changed_genes_per_strain)

    return (num_genes_changed_all / avg_changed_genes) if avg_changed_genes > 0 else 0

def simulate_null(direction_subset, num_simulations=1000):
    """Generate a null distribution by shuffling gene directions."""
    null_ratios = np.zeros(num_simulations)

    for i in range(num_simulations):
        shuffled_df = direction_subset.copy()
        for col in range(shuffled_df.shape[1]):
            shuffled_df.iloc[:, col] = np.random.permutation(shuffled_df.iloc[:, col].values)
        
        null_ratios[i] = compute_dice(shuffled_df)

    return null_ratios

def compute_dice_and_output(direction_df, output_file, num_simulations=1000):
    """Compute observed vs. simulated evolution ratios and write to file."""
    num_strains_total = len(direction_df.columns)
    
    with open(output_file, 'w') as f:
        f.write("Strain Combination\tObserved Ratio (%)\tSimulated Mean (%)\tSimulated Std Dev (%)\n")
        
        for num_strains in range(2, num_strains_total + 1):
            for strains in combinations(direction_df.columns, num_strains):
                direction_subset = direction_df[list(strains)]
                
                observed_ratio = compute_dice(direction_subset)
                null_ratios = simulate_null(direction_subset, num_simulations=num_simulations)

                mean_null = np.mean(null_ratios)
                std_null = np.std(null_ratios)

                f.write(f"{', '.join(strains)}\t{observed_ratio:.2f}\t{mean_null:.2f}\t{std_null:.2f}\n")
