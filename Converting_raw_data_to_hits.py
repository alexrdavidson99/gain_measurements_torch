import functools
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import argparse


@functools.lru_cache(maxsize=None)
def load_data(fname):
    _, pulse_min = np.loadtxt(fname, delimiter=",", unpack=True)
    return pulse_min

def parse_xy(f):
    return tuple((float(k) for k in f.stem.split("_")[1:]))


def extract_numbers_from_filename(filename):
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', filename)
    return [float(num) for num in numbers]



parser = argparse.ArgumentParser(description='Generate a heatmap from a CSV file.')
parser.add_argument('filename', type=str, help='Path to the CSV file')
args = parser.parse_args()

path_string = "C:/Users/lexda/Desktop/crosstalk_data_zip/Sweep_data_13_03_24/95_to_100_in_0.1_steps"
path_string = args.filename



DATA_DIR = Path(path_string)
start_postion = extract_numbers_from_filename(path_string)
start_postion_co = (start_postion[3] + (start_postion[4] - start_postion[3]) * 0.5)

print(start_postion_co)
print(DATA_DIR)

directory = DATA_DIR / 'hist'

# Create directory and its parents if they don't exist
os.makedirs(directory, exist_ok=True)


MIN_FILES_PREFIX = [f"F{i}" for i in range(1, 5)]
offsets = {'F1': -0.0001512, 'F2': -0.000430, 'F3': -0.000222, 'F4': +0.000002}

plt.figure(figsize=(14, 8))
for pref in MIN_FILES_PREFIX:

    files = DATA_DIR.glob(f"{pref}-sig*.csv")
    
    y_pts = []
    amplitude = []
    for f in files:
        x, y = parse_xy(f)
        hist_data = load_data(f)
        sig_min = load_data(f).mean()
        #dn_min = load_data(f.with_name(f.name.replace("sig", "dn"))).mean()
        
    
        counts, bins  = np.histogram(hist_data, bins=200)
        bins = bins[:-1]
        hist_data = {'bins': bins, 'counts': counts}
        df = pd.DataFrame(hist_data)
        
        df.to_csv( DATA_DIR/f'hist/histogram_{pref}_{start_postion_co+y:.1f}.txt', index=False)

    
        #data = sig_min - dn_min
        y_pts.append(start_postion_co+y)
        amplitude.append(-(sig_min))
    
    y_pts = np.array(y_pts)
    amplitude = np.array(amplitude)
    amplitude += offsets.get(pref, 0)
    
    sort_indices = y_pts.argsort()
    y_pts = y_pts[sort_indices]
    amplitude = amplitude[sort_indices]

    peaks, _ = find_peaks(amplitude, height=0.002, distance=10)

    
    plt.plot(y_pts, amplitude, label=pref)
    plt.plot(y_pts[peaks], amplitude[peaks], "x", label=f"{pref} peaks")
    plt.xlabel("position (mm)")
    plt.ylabel("mean Amplitude min pluse (V)")
    print(f"peaks for {pref}: {y_pts[peaks]}")
    plt.tick_params(axis='y',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=15)
   

    plt.legend()
plt.show()

# Directory containing the CSV files
folder_path = DATA_DIR

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Select a CSV file
file_name = csv_files[50]
file_path = os.path.join(folder_path, file_name)

# Read the CSV file using pandas
df = pd.read_csv(file_path, names=["x", "y"])
print(df)

# Extract the second row
second_row = df.iloc[150]

# Plot the histogram
counts, bins = np.histogram(df.y, bins=200)
print(counts[40])
counts, bins, bars = plt.hist(df.y, bins=200, alpha=0.7, color='blue', edgecolor='black', log=True)
print(counts[40])
bins = bins[:-1]
hist_data = {'bins': bins, 'counts': counts}
plt.plot(bins, counts)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Second Row')
plt.show()
print(counts)
df = pd.DataFrame(hist_data)
#df.to_csv(DATA_DIR /'histogram.txt', index=False)