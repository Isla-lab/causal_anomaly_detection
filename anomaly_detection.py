import time
import pandas as pd
import numpy as np
import os

from tigramite import plotting as tp
from matplotlib import pyplot as plt

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import time

import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# Constants
ALPHA = 0.05 # Significance level for ParCorr
PREFIX = "C:\\Users\\User\\tigramite\\tigramite\\tutorials\\causal_discovery\\" #path to model and data folder



def read_preprocess_data(path, timerange=[]):
    if TASK == 'swat':
        df = pd.read_csv(path, delimiter=";")
        df['Timestamp'] = df[' Timestamp'].str.strip()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p")
        df.set_index('Timestamp', inplace=True)
        columns_to_drop = [' Timestamp']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns or col == df.columns[0]], inplace=True)
        return df
    elif TASK == 'pepper':        
        df = pd.read_csv(path, delimiter=",")
        df['Timestamp'] = df['timestamp']
        df.set_index('Timestamp', inplace=True)
        columns_to_drop = ['timestamp']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns or col == df.columns[0]], inplace=True)
        return df

def run_pcmci(data, delay, link_assumptions=None):
    pcmci = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_max=delay, pc_alpha=ALPHA, link_assumptions=link_assumptions)

    return results


for TASK in ["pepper", "swat"]:
    print(TASK)
    end = 0
    start = 0
    has_ends = False
    tpos = []
    fpos = []
    tneg = []
    fneg = []

    causal_models = [f for f in os.listdir(PREFIX) if "npz" in f and TASK+"_normal_07" in f]
    f = np.load(PREFIX+causal_models[0], allow_pickle=True)
    val_matrix = f["val_matrix"]
    p_matrix = f["p_matrix"]
    var = list(f["var"])
    subsample = int(f['subsample'])
    delay = np.shape(val_matrix)[2] - 1
    nonconst = f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))
    normal_p_matrix = p_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > np.mean(abs(val_matrix)))

    #modify paths to dataset folders
    if TASK == "swat":
        attack_df = read_preprocess_data(PREFIX+"swat_dset/swat_csv/SWaT_Dataset_Attack_v0.csv")
        gt = attack_df['Normal/Attack']
        attack_df.drop(columns=['Normal/Attack'], inplace=True)
        timeranges=[['28/12/2015 01:10:10 PM', '28/12/2015 01:26:13 PM']] 
        attack_dfs = [attack_df]
        # print("DPIT301, tank 401 (8)")
        timeranges.append(['31/12/2015 01:45:19 AM', '31/12/2015 11:15:27 AM']) 
        attack_dfs.append(attack_df)
        # print("P302 (28)")
        timeranges.append(['29/12/2015 06:30:00 PM', '29/12/2015 06:42:00 PM']) 
        attack_dfs.append(attack_df)
        # print("MV101, LI1T101 (21)")
        timeranges.append(['29/12/2015 02:38:12 PM', '29/12/2015 02:50:08 PM']) 
        attack_dfs.append(attack_df)
        # print("MV303 (17)")
        timeranges.append(['30/12/2015 01:42:34 AM', '30/12/2015 01:54:10 AM']) 
        attack_dfs.append(attack_df)
        # print("P-602, DIT-301, DIT-301, MV-302 (23)")
        timeranges.append(['30/12/2015 10:01:50 AM', '30/12/2015 10:12:01 AM']) 
        attack_dfs.append(attack_df)
        # print("LIT-401, P-401, P402 (25)")
        timeranges.append(['30/12/2015 05:04:56 PM', '30/12/2015 05:29:00 PM']) 
        attack_dfs.append(attack_df)
        # print("P-101, LIT-301, P102 (26)")
        timeranges.append(['31/12/2015 01:17:08 AM', '31/12/2015 01:45:18 AM']) 
        attack_dfs.append(attack_df)
        # print("P-302, LIT-401 (27)")
        timeranges.append(['31/12/2015 03:47:40 PM', '31/12/2015 04:07:10 PM']) 
        attack_dfs.append(attack_df)
        # print("LIT-101, P-101, MV-201, MV101, LIT301 (30)")
        starts = [np.where(attack_df.index == timerange[0])[0][0] for timerange in timeranges]
        ends = [np.where(attack_df.index == timerange[1])[0][0] for timerange in timeranges]
        has_ends = True
        normal_df = read_preprocess_data(PREFIX+"swat_dset/swat_csv/SWaT_Dataset_Normal_v1.csv")
        normal_df.drop(columns=['Normal/Attack'], inplace=True)
    elif TASK == "pepper":
        normal_df = read_preprocess_data(PREFIX+"pepper_csv/normal.csv")
        attack_dfs = [read_preprocess_data(PREFIX+"pepper_csv/WheelsControl.csv")]
        attack_dfs.append(read_preprocess_data(PREFIX+"pepper_csv/JointControl.csv"))
        attack_dfs.append(read_preprocess_data(PREFIX+"pepper_csv/LedsControl.csv"))



    #PLOT CAUSAL GRAPH
    # pcmci = PCMCI(dataframe=pp.DataFrame(np.nan_to_num(normal_df.values[:, nonconst])), cond_ind_test=ParCorr())
    # # graph = pcmci.get_graph_from_pmatrix(p_matrix=normal_p_matrix, alpha_level=ALPHA, 
    # #         tau_min=0, tau_max=delay, link_assumptions=None)
    # normal_matrix[abs(normal_matrix) < 0.3] = 1 #remove weak links
    # graph = pcmci.get_graph_from_pmatrix(p_matrix=normal_matrix, alpha_level=0.99, 
    #         tau_min=0, tau_max=delay, link_assumptions=None)
    # tp.plot_graph(
    #     val_matrix=normal_matrix,
    #     graph=graph,
    #     var_names=f["var"][nonconst],
    #     link_colorbar_label='cross-MCI',
    #     node_colorbar_label='auto-MCI',
    #     show_autodependency_lags=False,
    #     arrow_linewidth=5,
    #     tick_label_size=10,
    #     link_label_fontsize=0
    # )
    # plt.show()
    #PLOT CAUSAL GRAPH
    
    #save normal coeffs
    normal_data = normal_df.values
    normal_data = normal_data[::subsample, nonconst]
    normal_data = np.nan_to_num(normal_data)
    indices = np.array(np.where(normal_matrix != 0))
    fine_coeffs = dict()
    for var in np.unique(indices[1,:]):
        var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
        var_indices.sort(key= lambda a : a[-1])
        stack_list = []
        max_delay = var_indices[-1][2]
        for el in var_indices:
            stack_list.append(normal_data[max_delay-el[2] : np.shape(normal_data)[0]-el[2], el[0]])
        stack_list.append(np.ones(np.shape(normal_data)[0]-max_delay))
        coeffs = np.linalg.lstsq(np.column_stack(stack_list), normal_data[max_delay:, var])[0][:-1]
        fine_coeffs[var] = coeffs


    #NORMAL OUAD
    #compute online coeffs
    err = dict()
    max_time = np.shape(normal_data)[0] - np.shape(normal_matrix)[2]
    norm_agg = np.zeros((max_time, len(np.unique(indices[1,:]))))
    for j in range(0, max_time):
        for i in range(len(np.unique(indices[1,:]))):
            var = np.unique(indices[1,:])[i]
            var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
            var_indices.sort(key= lambda a : a[-1])
            stack_list = []
            max_delay = var_indices[-1][2]
            for el in var_indices:
                stack_list.append(normal_data[max_delay-el[2] : j+np.shape(normal_matrix)[2]-el[2], el[0]])
            stack_list.append(np.ones(j+np.shape(normal_matrix)[2]-max_delay))
            coeffs = np.linalg.lstsq(np.column_stack(stack_list), normal_data[max_delay : j+np.shape(normal_matrix)[2], var])[0][:-1]
            if var not in err.keys():
                err[var] = np.zeros((max_time, len(var_indices)))
            err[var][j, :] = (coeffs - fine_coeffs[var])
            norm_agg[j,i] = np.linalg.norm(err[var][j, :])

    #sort by coeff error norm
    norms = np.linalg.norm(norm_agg[:,:], axis=0)
    norm_index_agg = np.argsort(norms)[::-1]
    indices_error = []
    for i in norm_index_agg:
        var = np.unique(indices[1,:])[i]
        var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
        norms = np.linalg.norm(err[var][:,:], axis=0)
        norm_index = np.argsort(norms)[::-1]
        #raise alarms
        for j in norm_index:
            thresh = np.linalg.norm(err[var][:,j])# / np.shape(err[var])[0]
            if np.linalg.norm(err[var][:,j]) > thresh:
                indices_error += list(np.where(abs(err[var][:,j]) > thresh)[0])
                dep_vars[f["var"][nonconst][var]].append(f["var"][nonconst][var_indices[j][0]])
            else:
                break

    fpos.append(len(np.unique(indices_error)))

    #ATTACK OUAD
    for q in range(len(attack_dfs)):
        #compute online coeffs
        if (q==0 and TASK=="swat") or TASK != "swat": #swat only has one data log: compute only once
            attack_data = attack_dfs[q].values
            attack_data = attack_data[::subsample, nonconst]
            attack_data = np.nan_to_num(attack_data)
            max_time = np.shape(attack_data)[0] - np.shape(normal_matrix)[2]
            norm_agg = np.zeros((max_time, len(np.unique(indices[1,:]))))
            err_attack = dict()
            for j in range(0, max_time):
                for i in range(len(np.unique(indices[1,:]))):    
                    var = np.unique(indices[1,:])[i]
                    var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
                    var_indices.sort(key= lambda a : a[-1])
                    max_delay = var_indices[-1][2]
                    start_time = time.time()
                    stack_list = []
                    for el in var_indices:
                        stack_list.append(attack_data[max_delay-el[2] : j+np.shape(normal_matrix)[2]-el[2], el[0]])
                    stack_list.append(np.ones(j+np.shape(normal_matrix)[2]-max_delay))
                    coeffs = np.linalg.lstsq(np.column_stack(stack_list), attack_data[max_delay : j+np.shape(normal_matrix)[2], var])[0][:-1]
                    if var not in err_attack.keys():
                        err_attack[var] = np.zeros((max_time, len(var_indices)))
                    err_attack[var][j, :] = (coeffs - fine_coeffs[var]) 
                    norm_agg[j,i] = np.linalg.norm(err_attack[var][j, :])

        start_index = 0
        end_index = -1
        if has_ends: #to isolate different anomalies in the unique log of swat
            start_index = starts[q]/subsample
            end_index = ends[q]/subsample
        #sort by coeff error norm
        norms = np.linalg.norm(norm_agg[int(start_index):int(end_index),:], axis=0)
        norm_index_agg = np.argsort(norms)[::-1]
        main_vars = [] #print this to identify broken causal children
        dep_vars = dict() #print this to identify broken causal parents
        indices_error = []
        for i in norm_index_agg:
            var = np.unique(indices[1,:])[i]
            var_indices = [indices[:,k] for k in range(np.shape(indices)[1]) if indices[1,k] == var]
            main_vars.append(f["var"][nonconst][var])
            norms = np.linalg.norm(err_attack[var][int(start_index):int(end_index),:], axis=0)
            norm_index = np.argsort(norms)[::-1]
            dep_vars[f["var"][nonconst][var]] = []
            #raise alarm
            for j in norm_index:
                thresh = np.linalg.norm(err[var][:,j])# / np.shape(err[var])[0]
                if np.linalg.norm(err_attack[var][int(start_index):int(end_index),j]) > thresh:
                    indices_error += list(np.where(abs(err_attack[var][int(start_index):int(end_index),j]) > thresh)[0])
                    dep_vars[f["var"][nonconst][var]].append(f["var"][nonconst][var_indices[j][0]])
                else:
                    break
            
            
        tpos.append(len(np.unique(indices_error)))
        if has_ends:
            fneg.append(np.shape(err_attack[var][int(start_index):int(end_index),j])[0] - tpos[-1])
        else:
            fneg.append(np.shape(err_attack[var][int(start_index):int(end_index),j])[0] - tpos[-1])

    print(TASK)
    print("PRECISION")
    print(np.sum(tpos) / (np.sum(tpos)+np.sum(fpos)))
    print("RECALL")
    print(np.sum(tpos) / (np.sum(tpos)+np.sum(fneg)))
    print("F1")
    print(2 * np.sum(tpos) / (2*np.sum(tpos)+np.sum(fneg)+np.sum(fpos)))
