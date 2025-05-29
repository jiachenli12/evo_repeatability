import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
from itertools import product
import seaborn as sns

## the code below takes a log2 fold change dataframe (log2(evolved/ancestor))
## this simulates the null distribution for each pair of parallel lines and combinations of >2 parallel lines

def assign_direction(logfc, df):
    """
    Assigns directions based on the logFC threshold.
    """
    return df.applymap(lambda x: 1 if x > logfc else (-1 if x < -logfc else 0))

def count_matching_directions(df):
    """
    Count the number of rows where all directions match (either all 1s or all -1s; all 0s doesn't count).
    """
    return np.sum(np.all(df != 0, axis=1) & (np.std(df, axis=1) == 0))

def simulate_parallel_genes_shuffle(df, num_simulations=1000):
    """
    Simulate the parallel gene evolution by shuffling directions.
    """
    counts = np.zeros(num_simulations, dtype=int)
    
    # fix the first column
    fixed_column = df.iloc[:, 0].values

    # shuffling
    for i in range(num_simulations):
        # Shuffle all columns except the first one
        shuffled_df = df.copy()
        for col in range(1, df.shape[1]):
            shuffled_df.iloc[:, col] = np.random.permutation(df.iloc[:, col].values)

        shuffled_df.iloc[:, 0] = fixed_column
        
        counts[i] = count_matching_directions(shuffled_df)
    
    return counts

def plot_simulation_results(simulated_counts, observed_count, title, ax):
    """
    Plotting: observed counts and distance in SDs.
    """
    mean_simulated = np.mean(simulated_counts)
    std_simulated = np.std(simulated_counts)
    D = (observed_count - mean_simulated) / std_simulated

    ax.hist(simulated_counts, bins=30, alpha=0.75)
    ax.axvline(x=observed_count, color='red', label=f'Observed: {observed_count}')
    ax.set_title(title)
    ax.set_xlabel("Number of genes evolved in the same direction")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.text(0.95, 0.85, f"D = {D:.2f} SD", transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right')
