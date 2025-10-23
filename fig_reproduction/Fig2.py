#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## raw data to plot

# first def
distance_data = {
    "Copepod in future environment": [24.59, 41.30, 21.84, 42.90, 62.88, 42.19],
    "Guppy in low-predation streams": [80.65],
    "Fly in hot temperature": [40.70, 38.05, 24.26, 24.87, 48.39, 20.47, 22.97, 20.60, 29.94, 44.15],
    "$\\it{E. coli}$ B REL1206 in 42°C": [43.88],
    "Yeast in NaCl": [17.62, 18.05, 39.81],
    "Yeast in H$_2$O$_2$": [61.80, 66.86, 65.79],
    "$\\it{E. coli}$ in NaCl": [46.14, 30.50, 28.47, 15.44, 24.74, 32.93, 20.72, 49.08, 29.21, 33.55],
    "$\\it{E. coli}$ in KCl": [13.97, 31.2, 8.64, 4.0, 25.55, 24.11, 7.34, 16.9, 3.44, 10.7],
    "$\\it{E. coli}$ in CoCl$_2$": [11.26, 40.60, 40.41, 35.61, 13.14, 12.20, 15.46, 47.45, 50.33, 41.66],
    "$\\it{E. coli}$ in Na$_2$CO$_3$": [39.57, 34.48, 32.89, 36.62, 45.18, 45.81, 46.54, 36.39, 37.88, 45.78],
    "$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)": [22.43, 27.81, 23.66, 24.17, 26.65, 24.97, 11.42, 23.40, 16.62, 20.40],
    "$\\it{E. coli}$ in Mal": [-0.05, 17.41, -0.08, 7.19, -0.11, 31.61, -0.08, -0.09, 5.59, -0.06],
    "$\\it{E. coli}$ in MCL": [22.29, 17.50, 17.74, 21.27, 21.74, 15.99, 21.30, 39.07, 27.73, 28.65],
    "$\\it{E. coli}$ in MG": [50.37, 44.91, 24.93, 51.73, 51.91, 27.99, 57.42, 29.81, 55.06, 30.06],
    "$\\it{E. coli}$ in BuOH": [27.08, 38.01, 29.21, 25.55, 29.87, 29.05, 38.37, 37.17, 21.43, 25.14],
    "$\\it{E. coli}$ in CPC": [3.7, -0.25, 16.15, 12.38, 24.46, 18.38, 17.31, 9.01, 13.57, 58.19],
    "$\\it{E. coli}$ in Cro": [47.46, 25.18, 48.25, 21.75, 29.93, 46.71, 21.36, 23.48, 43.12, 21.17],
    "Morning glory with herbicide": [28.65, 29.51, 48.89, 36.85, 35.42, 30.87, 25.47, 54.58, 30.87, 21.32, 20.55, 45.30,
                                    42.53, 36.57, 21.80, 22.26, 52.00, 51.25, 35.33, 27.55, 24.00, 25.90, 49.41, 30.93,
                                    29.98, 31.07, 30.50, 50.19],
    "$\\it{E. coli}$ K-12 in 42°C": [20.29, 18.93, 23.23, 18.35, 34.39, 32.38, 29.36, 20.02, 31.41, 32.79, 29.31, 35.81,
                                    14.66, 18.26, 19.43, 33.43, 24.45, 27.50, 37.19, 16.68, 18.97, 18.33, 35.73, 24.03,
                                    28.84, 22.49, 22.79, 29.25, 26.10, 32.02, 14.84, 22.26, 17.56, 35.95, 26.03, 28.63,
                                    29.18, 15.85, 26.86, 22.35, 19.51, 29.24, 16.29, 34.71, 22.14],
    "$\\it{E. coli}$ LTEE": [17.66, 19.3, 14.68, 16.32, 20.29, 13.86, 10.89, 18.76, 24.67, 10.98, 31.84, 16.88,
                           21.62, 15.66, 16.26, 17.26, 26.03, 17.25, 13.99, 12.89, 13.57, 17.2, 13.26, 13.67,
                           20.83, 16.67, 10.23, 19.43, 15.51, 14.74, 15.9, 17.35, 15.51, 20.7, 10.4, 14.28,
                           18.62, 15.2, 20.13, 16.0, 17.3, 11.99, 21.27, 14.71, 16.53, 16.49, 21.08, 14.98,
                           15.65, 13.38, 12.19, 13.53, 15.42, 17.87, 12.21],
    "$\\it{E. coli}$ in glycerol": [57.51, 34.36, 29.14, 32.74, 30.63, 63.63, 42.06, 36.07, 38.0, 35.91, 57.61, 56.43, 
                                   62.22, 60.14, 36.2, 63.02, 62.37, 31.32, 63.72, 33.28, 31.28],
    "$\\it{E. coli}$ in lactate (Fong et al. 2005)": [31.88, 42.16, 35.79, 36.5, 37.64, 50.65, 52.34, 61.83, 62.27, 59.03, 43.8, 56.08, 
                                                    58.81, 59.8, 52.12, 66.0, 64.01, 45.44, 66.98, 47.53, 47.1]
}

non_sig_values = {
    r"$\it{E. coli}$ in CPC": [-0.25],
    r"$\it{E. coli}$ in Mal": [-0.05, -0.08, -0.11, -0.09, -0.06],
}

deg_data = pd.DataFrame({
    'organism': [
        'Copepod in future environment','Guppy in low-predation streams','Fly in hot temperature',
        '$\\it{E. coli}$ B REL1206 in 42°C','Yeast in NaCl','Yeast in H$_2$O$_2$',
        '$\\it{E. coli}$ in NaCl','$\\it{E. coli}$ in KCl','$\\it{E. coli}$ in CoCl$_2$',
        '$\\it{E. coli}$ in Na$_2$CO$_3$','$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)',
        '$\\it{E. coli}$ in Mal','$\\it{E. coli}$ in MCL','$\\it{E. coli}$ in MG',
        '$\\it{E. coli}$ in BuOH','$\\it{E. coli}$ in CPC','$\\it{E. coli}$ in Cro',
        'Morning glory with herbicide','$\\it{E. coli}$ K-12 in 42°C','$\\it{E. coli}$ LTEE',
        '$\\it{E. coli}$ in glycerol','$\\it{E. coli}$ in lactate (Fong et al. 2005)'
    ],
    'Observation': [
        22.7,41.8,27.8,68.6,37.0,87.2,43.2,21.7,45.3,59.9,30.7,7.56,34.1,62.8,43.7,25.3,48.2,23.7,47.8,38.3,78.1,76.2
    ],
    'Random expectation': [
        2.67,0.26,0.801,8.28,6.65,19.4,0.48,0.53,0.61,1.17,0.32,0.17,0.54,0.87,0.65,0.26,0.49,3.36,13.0,17.6,44.6,10.1
    ]
})
ci_deg = np.array([
    [4.46,0.776],[0,0],[3.12,0.0749],[0,0],[7.33,0.405],[0.394,0.0130],[4.83,0.07],[4.38,0.05],
    [7.20,0.05],[2.21,0.03],[2.50,0.02],[3.56,0.03],[3.36,0.04],[5.34,0.02],[2.72,0.04],[7.06,0.06],
    [5.48,0.07],[1.22,0.139],[1.50,0.394],[0.772,0.387],[3.42,2.50],[3.95,0.649]
])

# second def
distance_data_dir = {
    "Copepod in future environment": [36.01, 49.17, 30.46, 59.30, 94.50, 59.41],
    "Guppy in low-predation streams": [102.98],
    "Fly in hot temperature": [56.44, 53.10, 29.33, 39.05, 69.55, 27.26, 37.27, 31.78, 41.57, 47.03],
    "$\\it{E. coli}$ B REL1206 in 42°C": [56.67],
    "Yeast in NaCl": [21.33, 13.66, 49.15],
    "Yeast in H$_2$O$_2$": [90.09, 90.42, 90.11],
    "$\\it{E. coli}$ in NaCl": [68.39, 49.89, 36.36, 24.18, 39.57, 58.45, 32.80, 46.39, 47.50, 40.14],
    "$\\it{E. coli}$ in KCl": [19.75, 46.44, 13.52, 5.16, 42.74, 33.86, 11.01, 25.38, 4.50, 16.37],
    "$\\it{E. coli}$ in CoCl$_2$": [15.71, 52.00, 50.47, 45.09, 18.46, 19.73, 22.71, 59.11, 50.77, 44.32],
    "$\\it{E. coli}$ in Na$_2$CO$_3$": [63.34, 48.07, 48.70, 53.80, 61.93, 62.69, 65.09, 49.20, 51.46, 61.69],
    "$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)": [30.05, 35.20, 38.24, 32.01, 32.78, 31.75, 15.08, 27.75, 19.73, 28.13],
    "$\\it{E. coli}$ in Mal": [-0.06, 25.41, -0.07, 8.71, -0.08, 22.34, -0.05, -0.05, 7.19, -0.03],
    "$\\it{E. coli}$ in MCL": [27.40, 26.93, 21.70, 28.12, 28.03, 20.13, 33.40, 59.18, 37.77, 37.42],
    "$\\it{E. coli}$ in MG": [57.67, 48.74, 33.44, 57.84, 55.13, 41.00, 58.91, 41.74, 52.25, 40.08],
    "$\\it{E. coli}$ in BuOH": [43.26, 52.56, 40.31, 36.10, 38.24, 50.22, 56.60, 50.68, 30.36, 35.92],
    "$\\it{E. coli}$ in CPC": [6.12, -0.15, 28.43, 17.13, 37.52, 25.88, 23.89, 12.88, 20.16, 74.29],
    "$\\it{E. coli}$ in Cro": [72.27, 38.88, 71.16, 30.65, 40.85, 65.82, 29.92, 35.08, 51.51],
    "Morning glory with herbicide": [28.61, 28.59, 56.48, 41.45, 37.70, 31.01, 25.90, 60.30, 35.04, 16.10, 13.00, 54.63,
                                     43.77, 41.23, 14.59, 16.47, 65.18, 51.69, 39.33, 27.49, 22.29, 25.47, 52.68, 25.01,
                                     28.80, 25.08, 26.54, 57.25],
    "$\\it{E. coli}$ K-12 in 42°C": [26.58, 23.68, 31.94, 22.84, 47.71, 46.65, 39.75, 24.46, 42.85, 38.65, 36.89, 45.93,
                                     19.25, 22.90, 25.01, 41.61, 33.43, 32.82, 48.03, 20.42, 23.33, 22.84, 40.12, 29.49,
                                     37.55, 28.93, 28.61, 37.55, 31.13, 44.95, 17.37, 27.82, 22.35, 45.46, 35.45, 39.65,
                                     38.54, 18.62, 36.37, 29.26, 24.05, 40.78, 18.85, 46.88, 27.48],
    "$\\it{E. coli}$ LTEE": [29.09, 31.08, 22.69, 26.83, 32.57, 17.67, 13.65, 27.68, 39.00, 17.68, 44.47, 21.97, 31.19,
                             27.13, 20.45, 21.10, 35.88, 27.48, 21.04, 16.80, 20.22, 27.40, 19.20, 18.91, 33.58, 26.71,
                             15.75, 29.22, 27.14, 19.86, 12.82, 28.28, 21.76, 31.53, 18.70, 13.51, 19.24, 21.13, 29.18,
                             23.24, 28.73, 17.59, 33.66, 23.06, 27.21, 20.99, 34.04, 17.96, 22.42, 19.47, 15.35, 16.17,
                             23.64, 27.64, 18.10],
    "$\\it{E. coli}$ in glycerol": [35.96, 14.42, 15.97, 14.19, 11.12, 37.08, 20.60, 17.49, 19.79, 13.20, 30.73, 22.25,
                                   26.76, 24.29, 19.32, 27.35, 26.20, 14.00, 30.76, 18.85, 17.42],
    "$\\it{E. coli}$ in lactate (Fong et al. 2005)": [24.63, 27.46, 22.42, 28.06, 25.69, 35.27, 35.22, 39.88, 38.49, 40.06,
                                                     28.24, 41.23, 40.99, 41.44, 36.83, 43.09, 41.93, 29.55, 42.55, 32.02, 34.04]
}
non_sig_values_dir = {
    r"$\it{E. coli}$ in CPC": [-0.15],
    r"$\it{E. coli}$ in Mal": [-0.06, -0.07, -0.08, -0.05, -0.05, -0.03]
}
dir_data = pd.DataFrame({
    'organism': [
        '$\\it{E. coli}$ K-12 in 42°C','$\\it{E. coli}$ B REL1206 in 42°C','$\\it{E. coli}$ in NaCl',
        '$\\it{E. coli}$ in KCl','$\\it{E. coli}$ in CoCl$_2$','$\\it{E. coli}$ in Na$_2$CO$_3$',
        '$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)','$\\it{E. coli}$ in Mal',
        '$\\it{E. coli}$ in MCL','$\\it{E. coli}$ in MG','$\\it{E. coli}$ in BuOH','$\\it{E. coli}$ in CPC',
        '$\\it{E. coli}$ in Cro','Yeast in NaCl','Yeast in H$_2$O$_2$','Copepod in future environment',
        'Fly in hot temperature','Guppy in low-predation streams','Morning glory with herbicide',
        '$\\it{E. coli}$ LTEE','$\\it{E. coli}$ in glycerol','$\\it{E. coli}$ in lactate (Fong et al. 2005)'
    ],
    'Observation': [
        45.6,68.0,43.2,21.7,45.3,60.0,27.9,7.6,34.1,62.8,43.7,25.3,48.9,32.4,87.2,22.5,25.7,41.8,20.9,32.7,41.3,38.7
    ],
    'Random expectation': [
        8.3,4.4,0.25,0.27,0.38,0.59,0.15,0.07,0.28,0.64,0.33,0.26,0.49,4.4,10.3,0.41,0.64,0.16,2.61,8.79,23.5,5.07
    ]
})
ci_dir = np.array([
    [1.72,0.202],[0,0],[4.83,0.03],[4.38,0.02],[7.20,0.04],[2.21,0.02],[2.11,0.02],[3.56,0.01],
    [3.36,0.02],[5.34,0.04],[2.72,0.03],[7.06,0.03],[5.55,0.04],[9.26,0.30],[0.394,0.0859],
    [4.49,0.623],[3.56,0.105],[0,0],[1.49,0.149],[0.934,0.194],[2.00,1.30],[1.92,0.318]
])

# third def
distance_data_mag = {
    "Copepod in future environment": [21.10, 39.44, 18.39, 17.58, 63.70, 22.87],
    "Guppy in low-predation streams": [99.45],
    "Fly in hot temperature": [44.01, 32.52, 11.54, 18.75, 37.32, 10.12, 12.19, 5.43, 19.29, 38.70],
    "$\\it{E. coli}$ B REL1206 in 42°C": [70.34],
    "Yeast in NaCl": [19.34, 10.04, 50.24],
    "Yeast in H$_2$O$_2$": [100.54, 99.81, 100.57],
    "$\\it{E. coli}$ in NaCl": [67.07, 45.36, 45.06, 24.84, 40.12, 52.01, 36.07, 51.36, 38.30, 37.83],
    "$\\it{E. coli}$ in KCl": [18.35, 43.21, 11.06, 6.37, 44.07, 40.62, 9.88, 27.30, 5.17, 15.29],
    "$\\it{E. coli}$ in CoCl$_2$": [16.15, 54.80, 53.12, 44.32, 17.43, 17.43, 17.38, 54.73, 53.76, 44.19],
    "$\\it{E. coli}$ in Na$_2$CO$_3$": [56.58, 49.62, 45.77, 50.68, 64.96, 59.34, 64.62, 48.61, 52.25, 64.15],
    "$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)": [28.66, 33.93, 30.81, 40.75, 35.20, 30.81, 14.41, 30.81, 16.33, 27.25],
    "$\\it{E. coli}$ in Mal": [-0.07, 28.66, -0.06, 15.78, -0.05, 31.61, -0.06, -0.06, 8.39, -0.08],
    "$\\it{E. coli}$ in MCL": [27.40, 26.24, 24.35, 32.19, 30.36, 25.85, 30.00, 62.04, 48.86, 41.75],
    "$\\it{E. coli}$ in MG": [57.62, 53.28, 34.70, 62.27, 55.44, 41.67, 62.03, 44.63, 58.82, 43.83],
    "$\\it{E. coli}$ in BuOH": [44.61, 60.53, 41.06, 38.23, 42.53, 48.99, 58.92, 69.06, 30.63, 40.23],
    "$\\it{E. coli}$ in CPC": [6.52, -0.18, 24.51, 18.34, 34.61, 27.03, 24.29, 13.94, 17.07, 70.11],
    "$\\it{E. coli}$ in Cro": [70.06, 39.53, 68.35, 29.70, 41.94, 66.25, 32.38, 35.75, 56.77, 30.03],
    "Morning glory with herbicide": [15.27, 14.90, 34.11, 24.74, 23.83, 17.59, 13.13, 34.99, 16.18, 5.15, 3.67, 37.24,
                                     25.26, 23.04, 5.03, 9.47, 39.42, 38.42, 22.81, 8.41, 10.57, 11.04, 35.80, 11.29,
                                     13.27, 9.17, 12.88, 32.11],
    "$\\it{E. coli}$ K-12 in 42°C": [12.40, 10.98, 16.44, 8.87, 35.26, 29.65, 27.59, 15.34, 26.19, 27.90, 19.72, 34.46,
                                     11.60, 11.09, 14.56, 29.11, 21.83, 15.97, 33.04, 6.01, 12.42, 8.72, 28.46, 19.30,
                                     26.22, 13.49, 17.35, 22.50, 17.11, 36.98, 8.10, 15.93, 12.23, 27.93, 26.73, 24.48,
                                     23.49, 11.03, 20.03, 18.09, 10.09, 26.01, 9.05, 43.16, 14.83],
    "$\\it{E. coli}$ LTEE": [18.59, 20.10, 14.76, 16.52, 19.89, 14.69, 10.70, 18.12, 25.13, 11.09, 31.32, 16.49, 22.04,
                             15.46, 16.07, 16.17, 26.41, 16.17, 13.85, 12.67, 13.06, 17.04, 12.94, 13.59, 20.41, 16.09,
                             10.41, 18.89, 15.49, 14.23, 15.86, 18.59, 15.18, 22.09, 10.29, 14.98, 18.13, 15.41, 20.03,
                             15.53, 16.85, 12.07, 20.31, 14.33, 17.25, 15.75, 21.26, 15.13, 15.54, 12.96, 12.06, 13.44,
                             15.14, 18.56, 11.80],
    "$\\it{E. coli}$ in glycerol": [26.34, 16.07, 16.94, 14.70, 11.73, 35.78, 23.82, 21.33, 24.61, 21.67, 24.57, 28.64,
                                   31.52, 31.13, 18.52, 31.79, 31.91, 13.99, 35.45, 18.04, 16.99],
    "$\\it{E. coli}$ in lactate (Fong et al. 2005)": [19.19, 22.48, 17.87, 23.68, 20.63, 26.90, 29.34, 36.51, 35.12,
                                                     35.37, 20.52, 32.92, 32.23, 35.30, 34.05, 38.13, 39.89, 22.99,
                                                     40.75, 25.05, 29.60]
}
non_sig_values_mag = {
    r"$\it{E. coli}$ in CPC": [-0.18],
    r"$\it{E. coli}$ in Mal": [-0.07, -0.06, -0.05, -0.06, -0.06, -0.08]
}
mag_data = pd.DataFrame({
    'organism': ['Guppy in low-predation streams','Copepod in future environment','Fly in hot temperature',
                 'Yeast in NaCl','Yeast in H$_2$O$_2$','$\\it{E. coli}$ B REL1206 in 42°C','$\\it{E. coli}$ in NaCl',
                 '$\\it{E. coli}$ in KCl','$\\it{E. coli}$ in CoCl$_2$','$\\it{E. coli}$ in Na$_2$CO$_3$',
                 '$\\it{E. coli}$ in Lac (Horinouchi et al. 2017)','$\\it{E. coli}$ in Mal','$\\it{E. coli}$ in MCL',
                 '$\\it{E. coli}$ in MG', '$\\it{E. coli}$ in BuOH', '$\\it{E. coli}$ in CPC','$\\it{E. coli}$ in Cro',
                 'Morning glory with herbicide','$\\it{E. coli}$ K-12 in 42°C','$\\it{E. coli}$ LTEE',
                 '$\\it{E. coli}$ in glycerol','$\\it{E. coli}$ in lactate (Fong et al. 2005)'],
    'Observation': [23.9,9.71,10.3,28.7,86.9,53.2,43.2,20.6,44.9,58.4,27.9,7.56,34.1,62.8,43.7,24.9,48.2,6.74,16.1,19.9,31.3,24.9],
    'Random expectation': [0.06,0.943,0.153,3.79,7.96,1.85,0.25,0.25,0.39,0.57,0.15,0.06,0.28,0.58,0.28,0.26,0.51,0.859,2.50,3.73,14.8,3.80]
})
ci_mag = np.array([
    [0,0],[2.67,0.385],[2.68,0.0358],[9.77,0.252],[0.426,0.0303],[0,0],[4.83,0.03],[4.28,0.01],[7.35,0.05],[2.24,0.02],
    [2.11,0.02],[3.56,0.02],[3.36,0.02],[5.34,0.04],[2.72,0.03],[7.09,0.03],[5.55,0.04],[0.665,0.0531],[0.947,0.0567],
    [0.752,0.0931],[2.00,0.826],[1.68,0.237]
])

# plotting
def _build_long_df(distance_data, non_sig_values):
    df = pd.DataFrame([(org, d) for org, ds in distance_data.items() for d in ds],
                      columns=["Organism", "Distance"])
    non_sig_values = non_sig_values or {}
    def _flag(row):
        vals = non_sig_values.get(row["Organism"], [])
        return "non-sig" if row["Distance"] in vals else "sig"
    df["flag"] = df.apply(_flag, axis=1)
    return df

def _prep_summary(summary_df, ci_array):
    s = summary_df.copy()
    s["Observation_frac"] = s["Observation"] / 100
    s["Random_frac"] = s["Random expectation"] / 100
    s = s.sort_values("Observation_frac", ascending=False).reset_index(drop=False)
    original_idx = s["index"].to_numpy()
    ci_sorted = ci_array[original_idx]
    s = s.reset_index(drop=True)
    s["y"] = s.index
    order = s["organism"].tolist()
    obs_err = ci_sorted[:, 0] / 100
    sim_err = ci_sorted[:, 1] / 100
    return s, order, obs_err, sim_err

def _format_axes(ax_left, ax_right, order, xlabel_left, xlabel_right):
    ax_left.set_yticks(range(len(order)))
    ax_left.set_yticklabels(order, fontsize=28)
    ax_left.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_left.set_xticklabels(['0','0.2','0.4','0.6','0.8'], fontsize=34)
    ax_left.set_xlabel(xlabel_left, fontsize=34)
    ax_left.tick_params(axis='x', labelsize=34, length=10, width=2)
    ax_left.tick_params(axis='y', labelsize=28, length=10)

    ax_right.set_xticks([0, 25, 50, 75, 100])
    ax_right.set_xticklabels(['0','25','50','75','100'], fontsize=34)
    ax_right.set_xlabel(xlabel_right, fontsize=30)
    ax_right.tick_params(axis='x', labelsize=34, length=10, width=2)
    ax_right.tick_params(axis='y', length=10)

    plt.subplots_adjust(left=0.45, right=0.95, wspace=0.15)
    sns.despine()

def make_panel(
    distance_data,               # dict: {organism: [values]}
    non_sig_values,              # dict: {organism: [non-sig values]}
    summary_df,                  # DataFrame: columns ['organism','Observation','Random expectation']
    ci_array,                    # np.array shape (n,2) aligned to summary_df
    obs_point_color="dodgerblue",
    strip_sig_color="dodgerblue",
    xlabel_left="Mean Dice's coefficient",
    xlabel_right=r"$\it{Z}$-score",
):
    df_long = _build_long_df(distance_data, non_sig_values)
    s, order, obs_err, sim_err = _prep_summary(summary_df, ci_array)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(34, 12), sharey=True)

    # left: paired means + errors
    for _, r in s.iterrows():
        ax1.plot([r["Random_frac"], r["Observation_frac"]], [r["y"], r["y"]],
                 color='gray', alpha=0.5, lw=1)
    ax1.scatter(s["Random_frac"], s["y"], marker='o', color='gray', s=300, zorder=3)
    ax1.scatter(s["Observation_frac"], s["y"], marker='o', color=obs_point_color, s=300, zorder=3)
    for i, r in s.iterrows():
        ax1.errorbar(r["Random_frac"], r["y"], xerr=sim_err[i],
                     fmt='none', ecolor='gray', elinewidth=3, capsize=12, zorder=2)
        ax1.errorbar(r["Observation_frac"], r["y"], xerr=obs_err[i],
                     fmt='none', ecolor=obs_point_color, elinewidth=3, capsize=12, zorder=2)

    # right: box + strip
    sns.boxplot(
        x="Distance", y="Organism", data=df_long, order=order,
        orient='h', fliersize=0, ax=ax2,
        boxprops=dict(facecolor='white', edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        showfliers=False
    )
    sns.stripplot(
        x="Distance", y="Organism",
        data=df_long[df_long.flag == "sig"], order=order,
        orient='h', size=8, alpha=0.7, color=strip_sig_color, jitter=True, ax=ax2
    )
    sns.stripplot(
        x="Distance", y="Organism",
        data=df_long[df_long.flag == "non-sig"], order=order,
        orient='h', size=8, alpha=0.7,
        edgecolor='grey', facecolor='grey', linewidth=1.5, jitter=True, ax=ax2
    )

    _format_axes(ax1, ax2, order, xlabel_left, xlabel_right)
    plt.show()

# Fig 2a,b
make_panel(
    distance_data=distance_data,
    non_sig_values=non_sig_values,
    summary_df=deg_data,
    ci_array=ci_deg,
    obs_point_color="dodgerblue",
    strip_sig_color="dodgerblue",
    xlabel_left="Mean Dice's coefficient",
    xlabel_right=r"$\it{Z}$-score",
)

# Fig 2c,d
make_panel(
    distance_data=distance_data_dir,
    non_sig_values=non_sig_values_dir,
    summary_df=dir_data,
    ci_array=ci_dir,
    obs_point_color="orange",
    strip_sig_color="orange",
    xlabel_left="Mean Dice's coefficient",
    xlabel_right=r"$\it{Z}$-score",
)

# Fig 2 e,f
make_panel(
    distance_data=distance_data_mag,
    non_sig_values=non_sig_values_mag,
    summary_df=mag_data,
    ci_array=ci_mag,
    obs_point_color="red",
    strip_sig_color="red",
    xlabel_left="Mean Dice's coefficient",
    xlabel_right=r"$\it{Z}$-score",
)
