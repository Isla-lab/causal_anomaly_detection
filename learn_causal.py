import time
import pandas as pd
import numpy as np

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import time

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# Constants
MAX_FREQ_COMPONENTS = 5
ALPHA = 0.05 # Significance level for ParCorr
PREFIX = "C:\\Users\\User\\tigramite\\tigramite\\tutorials\\causal_discovery\\" #path to data folder
STEP = 20

# Loads a specified CSV file, preprocesses it by selecting a subset of rows, removing near constant and NaN-only columns, formatting and indexing the timestamp, and eliminating unnecessary columns, returning the cleaned data in a pandas dataframe.
def read_preprocess_data(path, task):
    if task == 'swat':
        df = pd.read_csv(path, delimiter=";")
        df['Timestamp'] = df[' Timestamp'].str.strip()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p")
        ts = np.mean(df['Timestamp'].diff().dt.total_seconds())
        df.set_index("Timestamp", inplace=True)
        columns_to_drop = [' Timestamp', 'Normal/Attack']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns or col == df.columns[0]], inplace=True)
        return df, ts
    elif task == 'pepper':        
        df = pd.read_csv(path, delimiter=",")
        df['Timestamp'] = df['timestamp']
        ts = np.mean(df['Timestamp'].diff())*1e-3
        columns_to_drop = ['timestamp']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns or col == df.columns[0]], inplace=True)
        return df, ts



# Executes the PCMCI causal discovery algorithm
def run_pcmci(data, delay):
    data = np.nan_to_num(data)
    pcmci = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=delay, pc_alpha=ALPHA)

    return results








def main():
    for task in ["swat"]:
        print(task)
        #modify paths to dataset folders
        if task == 'swat':
            normal_df, ts = read_preprocess_data(PREFIX+"swat_dset/swat_csv/SWaT_Dataset_Normal_v1.csv", task)
        elif task == 'pepper':
            normal_df, ts = read_preprocess_data(PREFIX+"pepper_csv/normal.csv", task)

        columns = normal_df.columns
        normal_data = pp.DataFrame(np.nan_to_num(normal_df.values))

        #compute main frequencies
        frequencies = []
        for index in range(np.shape(normal_data.values[0])[1]):
            if any([el for el in normal_data.values[0][:,index] if int(el)!=el]): #consider frequencies of continuous variables
                w = np.fft.fft(normal_data.values[0][:,index])
                freqs = np.fft.fftfreq(len(w))
                mods = abs(w)
                max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
                main_freq = []
                for i in max_indices:
                    freq = freqs[i]
                    main_freq.append(freq)
                frequencies += main_freq

        #select most relevant frequencies (top 5%)
        sorted_freq = np.sort([el for el in frequencies if el>0])[::-1]
        for freq in sorted_freq:
            if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
                max_freq = freq
                sorted_freq = [s for s in sorted_freq if s <= max_freq]
                break

        #aubsample and filter, and remove nearly constant variables 
        normal_data.values[0] = normal_data.values[0][::max(1, int(np.floor(1/10/max_freq))), :]
        nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1]) if np.std(normal_data.values[0][:, idx]) > 0.01 * np.mean(normal_data.values[0][:, idx])]
        nonconst_data = normal_data.values[0][:, nonconst]
        for j in range(np.shape(nonconst_data)[1]):
            nonconst_data[:,j] /= (np.max(nonconst_data[:,j]) - np.min(nonconst_data[:,j])) + np.min(nonconst_data[:,j])
        
        #learn causal model
        delay = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
        start = time.time()
        normal_links = run_pcmci(nonconst_data, delay)
        elapsed = time.time() - start
        np.savez(PREFIX+task+"_normal", val_matrix=normal_links["val_matrix"], p_matrix=normal_links["p_matrix"], var=columns, subsample=max(1, int(np.floor(1/10/max_freq))), nonconst=nonconst, time=elapsed)


      



if __name__ == '__main__':
    main()