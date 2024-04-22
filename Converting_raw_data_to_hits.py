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

def extract_substring_from_path(input_path, base_folder):
    
    # Find the position of the base_folder in the input_path
    base_index = input_path.find(base_folder)
    
    if base_index != -1:
        # Extract the substring starting from the base_folder onwards
        extracted_substring = input_path[base_index:]
        return extracted_substring
    else:
        # Return an empty string if base_folder is not found in input_path
        return 'hi'

# Example usage:

base_folder = "/Sweep_data"

parser = argparse.ArgumentParser(description='Generate a heatmap from a CSV file.')
parser.add_argument('filename', type=str, help='Path to the CSV file')
args = parser.parse_args()

path_string = args.filename
Path_to_output_string_hist = 'C:/Users/lexda/VsProjects/gain_measurements_torch/Hist_outputs/'
Path_to_output_string = 'C:/Users/lexda/VsProjects/gain_measurements_torch/'

DATA_DIR = Path(path_string)

OUTPUT_DIR = Path(Path_to_output_string)
OUTPUT_DIR_HIST = Path(Path_to_output_string_hist)

start_postion = extract_numbers_from_filename(path_string)

start_postion_co = (start_postion[3] + (start_postion[4] - start_postion[3]) * 0.5)
extracted_string = extract_substring_from_path(path_string, base_folder)
print(extracted_string)  

print(start_postion_co)
print(DATA_DIR)

test = os.path.join(OUTPUT_DIR, extracted_string)
directory = os.path.join(Path_to_output_string_hist, extracted_string.replace('/', '_')) 
directory = Path(directory)

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
        
        df.to_csv( directory/f'histogram_{pref}_{start_postion_co+y:.1f}.txt', index=False)

    
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
plt.savefig(OUTPUT_DIR/'plots/plot.png')


