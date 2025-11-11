# Evolutionary repeatability
This repo contains python code and data for the paper: How repeatable is phenotypic evolution?


Please either clone this repo to run the code:

git clone https://github.com/jiachenli12/evo_repeatability.git



Directories in this repo:


fig_reproduction:
This directory contains code and all necessary data to reproduce main and supplementary figures presented in the paper. 

src:
This directory contains custom code to examine the three definitions of gene expression evoution repeatability.

1. deg_def1: the first definition of evolutionary repeatability. The script generates the null distribution of DEGs that are shared between parallel line pairs and >2 parallel lines. Observed number of shared DEGs and Z-scores are directly annotated on the plot.
   
2. same_direction_def2: the second definition of evolutionary repeatability

3. magnitude_def3: the third definition of evolutionary repeatability

4. dice_coefficient: compute dice's coefficent


Dependency: Python3
