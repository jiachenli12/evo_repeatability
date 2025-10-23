#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gff3_parser
from scipy.stats import ttest_ind, ranksums, mannwhitneyu, fisher_exact


# In[2]:


def assign_direction_general(logfc, df):
    return df.applymap(lambda x: 1 if x > logfc else (-1 if x < -logfc else 0))


# In[3]:


tf_data = pd.read_csv('/evo_repeatability/fig_reproduction/data/ecoli_TF_data.csv')


# In[4]:


lenski_deseq2_results = pd.read_csv('/evo_repeatability/fig_reproduction/data/lenski_deseq2_results.csv')


# In[5]:


confident_tf = (tf_data[tf_data['20)confidenceLevel'] !='?']).reset_index(drop=True)


# In[6]:


def expand_genes(token: str):
    m = re.match(r'^([a-z]+)', token)
    if not m:
        return [token]
    prefix = m.group(1)
    suffix = token[len(prefix):]
    return [prefix + letter for letter in suffix]

expanded_rows = []

clean = confident_tf.dropna(subset=['19)targetTuOrGene']).copy()
clean['19)targetTuOrGene'] = clean['19)targetTuOrGene'].astype(str)

for _, row in clean.iterrows():
    field = row['19)targetTuOrGene']
    if ':' not in field:
        continue  
    gene_str = field.split(':', 1)[1]
    gene_str = gene_str.strip('- ').strip()
    tokens = gene_str.split('-')
    
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok.lower() == 'none' or not re.match(r'^[A-Za-z0-9]+$', tok):
            continue
        
        for gene in expand_genes(tok):
            if gene and gene.lower() != 'none':
                expanded_rows.append({
                    'gene': gene,
                    'regulatorId': row['3)regulatorId']
                })

exploded_df = pd.DataFrame(expanded_rows)

gene_tf_count = (
    exploded_df
      .groupby('gene')['regulatorId']
      .nunique()
      .reset_index(name='num_tfs')
)


# In[7]:


exploded_df


# In[8]:


gff = gff3_parser.parse_gff3('/evo_repeatability/fig_reproduction/data/ecoli_b_rel1206_anno.gff',
                                   verbose = False,parse_attributes=True)


# In[9]:


gff['End'] = pd.to_numeric(gff['End'])
gff['Start'] = pd.to_numeric(gff['Start'])
gff['Length'] = gff['End'] - gff['Start'] + 1


# In[10]:


gff['Type'].unique()


# In[11]:


gff_filtered = gff[gff['Type']=='gene']


# In[12]:


gene_tf_map = pd.merge(gene_tf_count,gff_filtered,left_on='gene',right_on='Name',how='inner')


# In[13]:


gene_tf_map_non = pd.merge(gene_tf_count,gff_filtered,left_on='gene',right_on='Name',how='right')


# In[14]:


gene_tf_map_non['num_tfs'] = gene_tf_map_non['num_tfs'].fillna(0)


# In[15]:


import pandas as pd
import numpy as np
from functools import reduce

lenski_deseq2_results_rna = (
    lenski_deseq2_results
    .loc[lenski_deseq2_results['seqtype'] == 'rna']
    .reset_index(drop=True)
)

line_dfs_deseq2 = {
    line: grp.reset_index(drop=True)
    for line, grp in lenski_deseq2_results_rna.groupby('line')
}

def assign_directions(df):
    out = df[['target_id']].copy()
    out['direction'] = 0
    
    mask_up   = (df['padj'] < 0.05) & (df['log2FoldChange'] >  0)
    mask_down = (df['padj'] < 0.05) & (df['log2FoldChange'] <  0)
    
    out.loc[mask_up,   'direction'] =  1
    out.loc[mask_down, 'direction'] = -1
    
    return out

df_list = []
for line_name, df in line_dfs_deseq2.items():
    tmp = assign_directions(df)
    tmp = tmp.rename(columns={'direction': line_name})
    df_list.append(tmp)

combined_direction_df = reduce(
    lambda left, right: pd.merge(left, right, on='target_id', how='outer'),
    df_list
)


# In[16]:


merged_df = pd.merge(combined_direction_df,gene_tf_map_non,left_on = 'target_id',right_on = 'old_locus_tag',how='inner')


# In[17]:


merged_df = merged_df.iloc[:, list(range(1, 14)) + [16]]


# In[18]:


ara_cols = [c for c in merged_df.columns if c.startswith('Ara')]

merged_df['num_times_being_DEG'] = merged_df[ara_cols].abs().eq(1).sum(axis=1)
merged_df = merged_df[merged_df['num_times_being_DEG'] != 0]


# In[19]:


merged_df_copy = merged_df.copy()


# In[20]:


ara = merged_df[ara_cols]
num_nz = ara.ne(0).sum(axis=1)
sum_vals = ara.sum(axis=1)
merged_df_copy['num_reps_same_dir'] = np.where(
    sum_vals.abs() == num_nz,
    num_nz,
    0
)

dir_filtered = merged_df_copy.loc[merged_df_copy['num_reps_same_dir'] > 0].copy()


# In[21]:


import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr


# In[22]:


plt.figure(figsize=(12, 6))

# Get unique DEG groups
deg_levels = sorted(merged_df['num_times_being_DEG'].unique())
data_by_deg = [merged_df[merged_df['num_times_being_DEG'] == level]['num_tfs'].values for level in deg_levels]

# Plot violins
parts = plt.violinplot(data_by_deg, positions=deg_levels, showmeans=True, showmedians=False, showextrema=True)

for i, pc in enumerate(parts['bodies']):
    if i == 0:
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')
        pc.set_alpha(0.25)
    else:
        pc.set_facecolor('dodgerblue')
        pc.set_edgecolor('dodgerblue')
        pc.set_alpha(0.25)

for partname in ('cbars', 'cmins', 'cmaxes'):
    vp = parts[partname]
    vp.set_linewidth(1.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

for partname in ('cmeans',):
    vp = parts[partname]
    vp.set_linewidth(2.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

plt.xticks(deg_levels)
plt.xlabel('No. of replicates where a gene is a DEG', fontsize=20)
plt.ylabel('No. of TFs controlling a gene', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
sns.despine(top=True)

X_individual = sm.add_constant(merged_df['num_times_being_DEG'])
y_individual = merged_df['num_tfs']
model_individual = sm.OLS(y_individual, X_individual).fit()

unique_x = np.arange(merged_df['num_times_being_DEG'].min(), merged_df['num_times_being_DEG'].max() + 1)
X_pred = sm.add_constant(unique_x)
pred_line = model_individual.get_prediction(X_pred)
pred_line_df = pred_line.summary_frame(alpha=0.05)

plt.plot(unique_x, pred_line_df['mean'], linestyle='--', linewidth=0.75, color='black')
plt.fill_between(unique_x, pred_line_df['mean_ci_lower'], pred_line_df['mean_ci_upper'], color='black', alpha=0.1)

rho, pval = spearmanr(merged_df['num_times_being_DEG'], merged_df['num_tfs'])
p_val = r"$9.88 \times 10^{-13}$"
txt = f"$\\rho$ = {rho:.2f}\n$\\mathit{{P}}$ = {p_val}"
plt.text(0.67, 0.93, txt, transform=plt.gca().transAxes, verticalalignment='top', fontsize=20)
plt.title('$\it{E. coli}$ LTEE',fontsize=22)
plt.tight_layout()
plt.show()


# In[23]:


df_corr = merged_df[['num_tfs', 'num_times_being_DEG']].dropna()
from scipy.stats import pearsonr,spearmanr

pearson_r, pearson_p = pearsonr(df_corr['num_times_being_DEG'],df_corr['num_tfs'])
print(f"pearson r = {pearson_r:.3f}, pvalue = {pearson_p:.3g}")
spearman_rho, spearman_p = spearmanr(df_corr['num_times_being_DEG'],df_corr['num_tfs'])
print(f"spearman rho = {spearman_rho:.3f}, pvalue = {spearman_p:.3g}")


# In[24]:


mask = merged_df['gene_x'].isna()
placeholders = [f'placeholder_{i+1}' for i in range(mask.sum())]

merged_df.loc[mask, 'gene_x'] = placeholders


# In[25]:


merged_df


# In[26]:


ecoli_k12_gff = gff3_parser.parse_gff3('/evo_repeatability/fig_reproduction/data/ecoli_k12_anno.gff',
                                   verbose = False,parse_attributes=True)


# In[27]:


ecoli_k12_gff['End'] = pd.to_numeric(ecoli_k12_gff['End'])
ecoli_k12_gff['Start'] = pd.to_numeric(ecoli_k12_gff['Start'])
ecoli_k12_gff['Length'] = ecoli_k12_gff['End'] - ecoli_k12_gff['Start'] + 1


# In[28]:


ecoli_k12_gff_filtered = (ecoli_k12_gff[ecoli_k12_gff['Type']=='gene']).reset_index(drop=True)


# In[29]:


ecoli_42C_expr = pd.read_csv('/evo_repeatability/fig_reproduction/data/ecoli_42C_organized.csv')


# In[30]:


genes = pd.read_csv('/evo_repeatability/fig_reproduction/data/genes.txt')
ecoli_42C_expr = pd.concat([ecoli_42C_expr, genes], axis=1)


# In[31]:


ecoli_42C_expr


# In[32]:


p_cols = [col for col in ecoli_42C_expr.columns if col.startswith('WT')]
a_cols = [col for col in ecoli_42C_expr.columns if col.startswith('ALE')]
ecoli_42C_p = ecoli_42C_expr[p_cols]
ecoli_42C_a = ecoli_42C_expr[a_cols]
non_zero_mask = (ecoli_42C_p != 0).all(axis=1)
ecoli_42C_p = ecoli_42C_p[non_zero_mask]
ecoli_42C_a = ecoli_42C_a[non_zero_mask]
ecoli_42C_fc = np.log2(ecoli_42C_a.div(ecoli_42C_p.values, axis=0))
ecoli_42C_fc = pd.DataFrame(ecoli_42C_fc, index=ecoli_42C_a.index, columns=ecoli_42C_a.columns)


# In[33]:


ecoli_42C_expr = pd.read_csv('/evo_repeatability/fig_reproduction/data/ecoli_42C_organized.csv')
genes = pd.read_csv('/evo_repeatability/fig_reproduction/data/genes.txt')
ecoli_42C_expr['gene_id'] = genes['gene'].values

p_cols = [c for c in ecoli_42C_expr if c.startswith('WT')]
a_cols = [c for c in ecoli_42C_expr if c.startswith('ALE')]
p = ecoli_42C_expr[p_cols]
a = ecoli_42C_expr[a_cols]

mask = (p != 0).all(axis=1)
p = p[mask]; a = a[mask]

fc = np.log2(a.div(p.values, axis=0))

def assign_direction_general(threshold, df):
    return df.applymap(lambda x: 1 if x > threshold 
                                 else -1 if x < -threshold 
                                 else 0)
ecoli_42_2014_dir = assign_direction_general(2, fc)

ecoli_42_2014_dir = pd.concat([
    ecoli_42C_expr.loc[ecoli_42_2014_dir.index, ['gene_id']],
    ecoli_42_2014_dir
], axis=1)


# In[34]:


ecoli_42_2014_dir


# In[35]:


ecoli_k12_gff_filtered


# In[36]:


ecoli_42_2014_dir


# In[37]:


len(confident_tf['1)riId'].unique())


# In[38]:


confident_tf['20)confidenceLevel']


# In[39]:


confident_tf


# In[40]:


ecoli_42_2014_dir


# In[41]:


gene_tf_count


# In[42]:


ecoli_42_2014_tfs = pd.merge(gene_tf_count,ecoli_42_2014_dir,left_on='gene',right_on='gene_id',how='right')


# In[43]:


ecoli_42_2014_tfs['num_tfs'] = ecoli_42_2014_tfs['num_tfs'].fillna(0)


# In[44]:


ecoli_42_2014_tfs


# In[45]:


ale_cols = [c for c in ecoli_42_2014_tfs if c.startswith('ALE')]
ecoli_42_2014_tfs ['num_times_being_DEG'] = ecoli_42_2014_tfs [ale_cols].abs().eq(1).sum(axis=1)
ecoli_42_2014_tfs  = ecoli_42_2014_tfs [ecoli_42_2014_tfs ['num_times_being_DEG'] != 0]


# In[46]:


ecoli_42_2014_tfs 


# In[47]:


df_corr_2014 = ecoli_42_2014_tfs[['num_tfs', 'num_times_being_DEG']].dropna()

# pearson_r, pearson_p = pearsonr(df_corr_2014['num_times_being_DEG'],df_corr_2014['num_tfs'])
# print(f"pearson r = {pearson_r:.3f}, pvalue = {pearson_p:.3g}")
spearman_rho, spearman_p = spearmanr(df_corr_2014['num_times_being_DEG'],df_corr_2014['num_tfs'])
print(f"spearman rho = {spearman_rho:.3f}, pvalue = {spearman_p:.3g}")


# In[48]:


ale_cols = [c for c in ecoli_42_2014_tfs if c.startswith('ALE')]
ale_df = ecoli_42_2014_tfs[ale_cols]

num_nz = ale_df.ne(0).sum(axis=1)

sum_vals = ale_df.sum(axis=1)

ecoli_42_2014_tfs['num_reps_same_dir'] = np.where(
    sum_vals.abs() == num_nz,
    num_nz,
    0
)

ecoli_42_2014_tfs_dir = ecoli_42_2014_tfs[ecoli_42_2014_tfs['num_reps_same_dir'] > 0].copy()


# In[49]:


ecoli_42_2014_tfs_dir


# In[50]:


p_formatted = r"$6.42 \times 10^{-3}$"


# In[51]:


plt.figure(figsize=(10, 6))
deg_levels = sorted(ecoli_42_2014_tfs['num_times_being_DEG'].unique())
data_by_deg = [ecoli_42_2014_tfs[ecoli_42_2014_tfs['num_times_being_DEG'] == level]['num_tfs'].values for level in deg_levels]

parts = plt.violinplot(data_by_deg, positions=deg_levels, showmeans=True, showmedians=False, showextrema=True)

for i, pc in enumerate(parts['bodies']):
    if i == 0:
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')
        pc.set_alpha(0.25)
    else:
        pc.set_facecolor('dodgerblue')
        pc.set_edgecolor('dodgerblue')
        pc.set_alpha(0.25)

# Color other elements inside the violins
for partname in ('cbars', 'cmins', 'cmaxes'):
    vp = parts[partname]
    vp.set_linewidth(1.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

for partname in ('cmeans',):
    vp = parts[partname]
    vp.set_linewidth(2.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

plt.xticks(deg_levels)
plt.xlabel('No. of replicates where a gene is a DEG', fontsize=20)
plt.ylabel('No. of TFs controlling a gene', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
sns.despine(top=True)

X_individual = sm.add_constant(ecoli_42_2014_tfs['num_times_being_DEG'])
y_individual = ecoli_42_2014_tfs['num_tfs']
model_individual = sm.OLS(y_individual, X_individual).fit()

unique_x = np.arange(ecoli_42_2014_tfs['num_times_being_DEG'].min(), ecoli_42_2014_tfs['num_times_being_DEG'].max() + 1)
X_pred = sm.add_constant(unique_x)
pred_line = model_individual.get_prediction(X_pred)
pred_line_df = pred_line.summary_frame(alpha=0.05)

plt.plot(unique_x, pred_line_df['mean'], linestyle='--', linewidth=0.75, color='black')
plt.fill_between(unique_x, pred_line_df['mean_ci_lower'], pred_line_df['mean_ci_upper'], color='black', alpha=0.1)

rho, pval = spearmanr(ecoli_42_2014_tfs['num_times_being_DEG'], ecoli_42_2014_tfs['num_tfs'])
txt = f"$\\rho$ = {rho:.2f}\n$\\mathit{{P}}$ = {p_formatted}"
plt.text(0.7, 0.96, txt, transform=plt.gca().transAxes, verticalalignment='top', fontsize=20)
plt.ylim(None, 20)
plt.title('$\it{E. coli}$ K-12 in 42°C', fontsize=22)
plt.tight_layout()
plt.show()


# In[52]:


for i in range(1, 11):
    count = len(ecoli_42_2014_tfs[ecoli_42_2014_tfs['num_times_being_DEG'] == i])
    print(f'num_times_being_DEG == {i}: {count}')


# In[53]:


total = sum(len(ecoli_42_2014_tfs[ecoli_42_2014_tfs['num_times_being_DEG'] == i]) for i in range(5, 11))
total


# In[54]:


total = sum(len(ecoli_42_2014_tfs[ecoli_42_2014_tfs['num_times_being_DEG'] == i]) for i in range(2, 4))
total


# In[55]:


ecoli_42_2014_tfs


# In[56]:


ecoli_11_envs_p = pd.read_excel('/evo_repeatability/fig_reproduction/data/ecoli_11_envs_pState.xlsx')
ecoli_11_envs_a = pd.read_excel('/evo_repeatability/fig_reproduction/data/ecoli_11_envs_aState.xlsx')


# In[57]:


ecoli_11_envs_p 


# In[58]:


ecoli_11_envs_p = ecoli_11_envs_p.rename(columns={'CoCl': 'CoCl2', 'SoC': 'Na2CO3'})
ecoli_11_envs_p = ecoli_11_envs_p.drop(columns=['No stress'])


# In[59]:


fc = {}

for env in ecoli_11_envs_p.columns:
    if env == 'name':
        continue

   
    replicate_cols = [col for col in ecoli_11_envs_a.columns if col.startswith(env)]

    # start your result DataFrame with the 'name' column
    env_result = ecoli_11_envs_p[['name']].copy()

    # now compute each replicate's log2(a/p) and stick it in
    for col in replicate_cols:
        env_result[col] = np.log2(ecoli_11_envs_a[col] / ecoli_11_envs_p[env])

    # store it
    fc[env] = env_result

# now fc['NaCl'] (etc.) will have a first column named 'name'


# In[60]:


fc


# In[61]:


gene_tf_count


# In[62]:


def assign_direction(logfc, df):
    """
    Assigns directions based on the logFC threshold.
    """
    return df.applymap(lambda x: 1 if x > logfc else (-1 if x < -logfc else 0))


logfc = 0.5
direction_fc = {}

for env, df in fc.items():
    df_dir = df.copy()
    numeric_cols = [c for c in df.columns if c != 'name']
    df_dir[numeric_cols] = assign_direction(logfc, df[numeric_cols])
    direction_fc[env] = df_dir


# In[63]:


direction_fc


# In[64]:


gene_tf_count


# In[65]:


final_results = {}

for env, df_dir in direction_fc.items():
    merged = df_dir.merge(
        gene_tf_count,
        left_on='name',
        right_on='gene',
        how='left'
    )
    tf_cols = gene_tf_count.columns.difference(['name'])
    merged[tf_cols] = merged[tf_cols].fillna(0)

    deg_cols = [c for c in merged.columns if c not in ['name', *tf_cols]]
    merged['num_times_being_DEG'] = merged[deg_cols].abs().eq(1).sum(axis=1)

    filtered = merged[merged['num_times_being_DEG'] != 0].copy()
    
    final_results[env] = filtered


# In[66]:


final_results['NaCl']


# In[67]:


corr_summary = []

for env, df in final_results.items():
    df_corr = df[['num_tfs', 'num_times_being_DEG']].dropna()
    pearson_r, pearson_p = pearsonr(df_corr['num_times_being_DEG'], df_corr['num_tfs'])
    spearman_rho, spearman_p = spearmanr(df_corr['num_times_being_DEG'], df_corr['num_tfs'])

    print(f"[{env}] pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}")
    print(f"[{env}] spearman rho = {spearman_rho:.3f}, p = {spearman_p:.3g}\n")
    
    corr_summary.append({
        'env': env,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p
    })

corr_df = pd.DataFrame(corr_summary)


# In[68]:


mg = final_results['MG']


# In[69]:


final_results['NaCl']


# In[70]:


mg_p_val = r"$1.71 \times 10^{-3}$"


# In[71]:


#plot
plt.figure(figsize=(7, 6))

# Get unique DEG groups
deg_levels = sorted(mg['num_times_being_DEG'].unique())
data_by_deg = [mg[mg['num_times_being_DEG'] == level]['num_tfs'].values for level in deg_levels]

# Plot violins
parts = plt.violinplot(data_by_deg, positions=deg_levels, showmeans=True, showmedians=False, showextrema=True)

# Color violins: first one grey, others default (blueish)
for i, pc in enumerate(parts['bodies']):
    if i == 0:
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')
        pc.set_alpha(0.25)
    else:
        pc.set_facecolor('dodgerblue')
        pc.set_edgecolor('dodgerblue')
        pc.set_alpha(0.25)

# Color other elements inside the violins
for partname in ('cbars', 'cmins', 'cmaxes'):
    vp = parts[partname]
    vp.set_linewidth(1.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

for partname in ('cmeans',):
    vp = parts[partname]
    vp.set_linewidth(2.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

plt.xticks(deg_levels)
plt.xlabel('No. of replicates where a gene is a DEG', fontsize=20)
plt.ylabel('No. of TFs controlling a gene', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
sns.despine(top=True)

# -----------------------------------------------------------------------
# --- Linear regression line & confidence interval ----------------------

X_individual = sm.add_constant(mg['num_times_being_DEG'])
y_individual = mg['num_tfs']
model_individual = sm.OLS(y_individual, X_individual).fit()

unique_x = np.arange(mg['num_times_being_DEG'].min(), mg['num_times_being_DEG'].max() + 1)
X_pred = sm.add_constant(unique_x)
pred_line = model_individual.get_prediction(X_pred)
pred_line_df = pred_line.summary_frame(alpha=0.05)

plt.plot(unique_x, pred_line_df['mean'], linestyle='--', linewidth=0.75, color='black')
plt.fill_between(unique_x, pred_line_df['mean_ci_lower'], pred_line_df['mean_ci_upper'], color='black', alpha=0.1)

# -----------------------------------------------------------------------
# --- Spearman correlation annotation -----------------------------------

rho, pval = spearmanr(mg['num_times_being_DEG'], mg['num_tfs'])
txt = f"$\\rho$ = {rho:.2f}\n$\\mathit{{P}}$ = {mg_p_val}"
plt.text(0.45, 0.93, txt, transform=plt.gca().transAxes, verticalalignment='top', fontsize=20)
plt.title('$\it{E. coli}$ in MG', fontsize=22)

plt.tight_layout()
plt.show()


# In[72]:


deg_dfs = [
    df[['name', 'num_times_being_DEG']]
    for df in final_results.values()
]

# 2) stack vertically into one long DataFrame
all_deg = pd.concat(deg_dfs, ignore_index=True)

deg_summary = (
    all_deg
    .groupby('name', as_index=False)
    .agg(total_DEG=('num_times_being_DEG', 'sum'))
)


# In[73]:


filtered_results = {}

for cond, df in final_results.items():
    # 1) Work on a copy
    df = df.copy()
    
    # 2) Identify just the replicate columns (e.g. "NaCl-1", "NaCl-2", …)
    rep_cols = [c for c in df.columns if c.startswith(f"{cond}-")]
    
    # 3) Count non‐zero calls per row, and sum of values per row
    num_nz   = df[rep_cols].ne(0).sum(axis=1)
    sum_vals = df[rep_cols].sum(axis=1)
    
    # 4) Create your 'num_reps_same_dir' column
    df['num_reps_same_dir'] = np.where(
        sum_vals.abs() == num_nz,   # if all non‐zeros agree in sign
        num_nz,                      #   then record how many
        0                            # else zero
    )
    
    # 5) Keep only the rows with at least one “same‐direction” replicate
    filtered_results[cond] = df[df['num_reps_same_dir'] > 0].copy()


# In[74]:


dir_dfs = [
    df[['name', 'num_reps_same_dir']]
    for df in filtered_results.values()
]

# 2) stack vertically into one long DataFrame
all_dir = pd.concat(dir_dfs, ignore_index=True)


# In[75]:


dir_summary = (
    all_dir
    .groupby('name', as_index=False)
    .agg(total_dir=('num_reps_same_dir', 'sum'))
)


# In[76]:


dir_summary


# In[77]:


deg_summary


# In[78]:


tf_dfs = [
    df[['name', 'num_tfs']]
    for df in final_results.values()
]
all_tf = pd.concat(tf_dfs, ignore_index=True)

# drop duplicates so you have exactly one num_tfs per gene
tf_summary = all_tf.drop_duplicates(subset='name')

# 5) merge the two summaries
final = deg_summary.merge(tf_summary, on='name')


# In[79]:


final_dir = dir_summary.merge(tf_summary, on='name')


# In[80]:


final_dir


# In[81]:


final


# In[82]:


final['deg_bin'] = final['total_DEG'].clip(upper=16)


# In[83]:


final


# In[84]:


r_pearson, p_pearson   = pearsonr(final['deg_bin'], final['num_tfs'])
r_spearman, p_spearman = spearmanr(final['deg_bin'], final['num_tfs'])

print(f"per gene: Pearson r = {r_pearson:.3f}, p = {p_pearson:.3g}")
print(f"per gene: Spearman rho = {r_spearman:.3f}, p = {p_spearman:.3g}")


# In[85]:


final_p = r"$4.36 \times 10^{-5}$"


# In[86]:


plt.figure(figsize=(10, 6))

deg_levels = sorted(final['deg_bin'].unique())
data_by_deg = [final[final['deg_bin'] == level]['num_tfs'].values for level in deg_levels]

parts = plt.violinplot(data_by_deg, positions=deg_levels, showmeans=True, showmedians=False, showextrema=True)

for i, pc in enumerate(parts['bodies']):
    if i == 0:
        pc.set_facecolor('grey')
        pc.set_edgecolor('grey')
        pc.set_alpha(0.25)
    else:
        pc.set_facecolor('dodgerblue')
        pc.set_edgecolor('dodgerblue')
        pc.set_alpha(0.25)

for partname in ('cbars', 'cmins', 'cmaxes'):
    vp = parts[partname]
    vp.set_linewidth(1.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

for partname in ('cmeans',):
    vp = parts[partname]
    vp.set_linewidth(2.5)
    vp.set_color(['grey'] + ['dodgerblue'] * (len(deg_levels) - 1))

xtick_labels = [str(l) for l in deg_levels[:-1]] + ['\u226516']

plt.xticks(
    ticks=deg_levels,
    labels=xtick_labels,
    ha='center',
    fontsize=18
)
plt.xlabel('No. of replicates where a gene is a DEG', fontsize=20)
plt.ylabel('No. of TFs controlling a gene', fontsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=18)
sns.despine(top=True)

X_individual = sm.add_constant(final['deg_bin'])
y_individual = final['num_tfs']
model_individual = sm.OLS(y_individual, X_individual).fit()

unique_x = np.arange(final['deg_bin'].min(), final['deg_bin'].max() + 1)
X_pred = sm.add_constant(unique_x)
pred_line = model_individual.get_prediction(X_pred)
pred_line_df = pred_line.summary_frame(alpha=0.05)

plt.plot(unique_x, pred_line_df['mean'], linestyle='--', linewidth=0.75, color='black')
plt.fill_between(unique_x, pred_line_df['mean_ci_lower'], pred_line_df['mean_ci_upper'], color='black', alpha=0.1)

rho, pval = spearmanr(final['deg_bin'], final['num_tfs'])
txt = f"$\\rho$ = {rho:.2f}\n$\\mathit{{P}}$ = {final_p}"
plt.text(0.7, 0.93, txt, transform=plt.gca().transAxes, verticalalignment='top', fontsize=20)
plt.title('$\it{E. coli}$ in 11 harsh environments', fontsize=22)
plt.ylim(None, 25)
plt.tight_layout()
plt.show()


# In[87]:


replicate_cols = [col for col in combined_direction_df.columns if col != 'target_id']
repeat_counts = (combined_direction_df[replicate_cols].abs() >= 1).sum(axis=1)
repeatable = repeat_counts >= 2

repeat_counts = pd.Series(repeat_counts.values, index=combined_direction_df['target_id'], name='num_replicates_changed')
repeatable = pd.Series(repeatable.values, index=combined_direction_df['target_id'], name='is_repeatable')


# In[88]:


hist_obs = repeat_counts.value_counts().sort_index()

k_plot = np.arange(0, 12)
obs_counts = hist_obs.reindex(k_plot, fill_value=0).values

lambda_obs = repeat_counts.mean() #lambda is the mean of nums of environments in which a gene is repeatable
N = len(repeat_counts)

sim = np.random.poisson(lam=lambda_obs, size=N)
sim_counts = pd.Series(sim).value_counts().reindex(k_plot, fill_value=0).values

bar_width = 0.4
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(k_plot - bar_width/2, obs_counts, width=bar_width, color='dodgerblue',label='Observation')
ax.bar(k_plot+ bar_width/2, sim_counts, width=bar_width, color='grey',label= 'Poisson expectation')

ax.set_xticks(k_plot)
ax.set_xlabel('No. of replicates where a gene is a DEG', fontsize=20)
ax.set_ylabel('No. of genes',fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.ylim(0,2500)
plt.text(0.45, 0.7, r'$\mathit{P} < 0.001$', 
         verticalalignment='top', transform=ax.transAxes,fontsize=25)
sns.despine(top=True)
plt.legend(frameon=False, fontsize=16)
plt.show()


# In[89]:


from scipy.stats import chi2

obs_var = np.var(obs_counts)  
poisson_var = lambda_obs
n = len(repeat_counts)

chi_squared_stat = (n-1)*obs_var/poisson_var

p_value = chi2.sf(chi_squared_stat, df=n-1)

print(f"chi-squared statistic: {chi_squared_stat:.3f}")
print(f"p-value: {p_value:.5f}")

