import os
import re
import gc
import json
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
import xml.etree.ElementTree as ET
from collections import defaultdict
from importlib import import_module
from statsmodels.stats.multitest import multipletests

# THIS SCRIPT READS IN ALL THE RESULTS, FORMATTED AS NIFTI FILES WHERE THE VALUE OF EACH VOXEL
# CORRESPONDS TO THE CORRELATION COEFFICIENT BETWEEN THE PREDICTED AND ACTUAL SIGNAL.
# THOSE RESULTS ARE ORGANIZED IN TWO HEAVILY NESTED (ARGUABLY TOO MUCH SO) DICTIONARIES WHERE VALUES
# ARE THE TOP 10,000 CORRELATION COEFFICIENTS, AND SEPARATELY, THE SCHAEFER PARCEL NUMBERS ASSOCIATED WITH 
# THOSE CORRELATION COEFFICIENTS. 

seed = 888
fdr_alpha = 0.05
npermutations = 1000
fir = import_module('6_fir')
perm = import_module('_permutation_testing')
rng = np.random.default_rng(seed)
conditions = ['recursion', 'iteration', 'prose', 'list']

# read in 2d mni mask
mask = nib.load("misc/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load("misc/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx] # contains the schaefer parcel numbers
cortex_vx = np.where(atlas_only_brain != 0)[0]
parcel_nums = atlas_only_brain[cortex_vx]
hemis = ['Left' if ((parcel <= 200) & (parcel > 0)) else 'Right' for parcel in parcel_nums]

# Loading in atlas and labels for harvard-oxford atlas regions
hox_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
hox = nib.load(hox_data['filename']).get_fdata()
hox_vec = hox.flatten()
hox_only_brain = hox_vec[brain_idx]
hox_nums = hox_only_brain[cortex_vx] # Obtaining voxels in Harvard Oxford atlas that are in the schaefer atlas.
                                     #      The voxels in the Schaefer atlas that correspond to empty space in HarvOx atlas (values of 0) will get filtered out when we match to region names.

hox_label_path = "misc/HarvardOxford-Cortical.xml"
hox_labels = [label.text for label in ET.parse(hox_label_path).getroot().iter('label')] # indices are the region numbers - need to add one though

# Making empty templates to save output
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)

def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
        
        
def map_to_schaefer_regions(voxel_idx):
    top_parcels = parcel_nums[voxel_idx]
    top_parcels = np.array([int(parcel) for parcel in top_parcels])
    return top_parcels

def in_harvox_mask(voxel_idx):
    # True where a voxel (indexed into the schaefer/cortex space) also falls inside
    # the Harvard-Oxford cortical atlas; False means it should be ignored for HarvOx comparisons
    return hox_nums[np.asarray(voxel_idx)] != 0

def map_to_HarvOx_regions(voxel_idx):
    top_hox_idx = [int(idx) for idx in voxel_idx if hox_nums[idx] != 0] # finding locations for top 10k voxels 
    top_regions = hox_nums[top_hox_idx] # mapping to HarvOx region numbers
    top_regions = np.array([int(roi) for roi in top_regions])

    # finding corresponding hemispheres for voxel locations. Need to index by top_hox_idx to properly map to
    # correct locations in hemis list
    top_regions = [f"{hemis[top_hox_idx[i]]} {hox_labels[roi-1]}" for i,roi in enumerate(top_regions)]

    return top_regions

# def find_cutoff(vec, threshold = 10**4):
def find_cutoff(vec, threshold = 6525): # the average number of voxels that are modeled significantly better than chance
    
    # I can find the threshold point based on sorting, then keep everything in the same place
    copy = vec.copy()
    copy.sort()
    cutoff = copy[-threshold-1]
    return cutoff

def zscore_cols(mat):
    mat = np.asarray(mat, dtype=float)
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd

def permutation_test(Presp, pred, nperms=1000, min_shift=None,
                            two_sided=False, seed=None):
    """
    Circular-shift permutation test for voxelwise prediction correlations.

    Presp: (T, V) actual test-set responses
    pred:  (T, V) predicted test-set responses (np.dot(emb_test, weights))
    nperms: number of circular shifts to draw
    min_shift: smallest |shift| (in TRs) allowed, so a permutation can't
               land close enough to zero to trivially recover the true
               alignment; defaults to 5% of T
    two_sided: if True, compare |corr| instead of signed corr

    Returns:
      obs_corrs: (V,) observed correlation per voxel
      pvals: (V,) empirical p-value per voxel, P(null >= observed)
    """
    # rng = np.random.default_rng(seed)
    T, V = Presp.shape
    if min_shift is None:
        min_shift = max(1, T // 20)

    actual_z = zscore_cols(Presp)
    pred_z = zscore_cols(pred)

    obs_corrs = (actual_z * pred_z).mean(axis=0)
    obs_stat = np.abs(obs_corrs) if two_sided else obs_corrs

    exceed_count = np.zeros(V)
    for _ in range(nperms):
        shift = rng.integers(min_shift, T - min_shift)
        null_corr = (actual_z * np.roll(pred_z, shift, axis=0)).mean(axis=0)
        null_stat = np.abs(null_corr) if two_sided else null_corr
        exceed_count += (null_stat >= obs_stat)

    pvals = (1 + exceed_count) / (nperms + 1)  # +1/+1 correction avoids p=0
    return obs_corrs, pvals


def permutation_test_wrapper(filepath, participant, condition):
    embedding_path = f"output/stimulus_matrices/{participant}-{condition}-stimulus_matrix.pkl"
    
    print(f"Participant {participant}: loading test data")
    _, fmri_test, _, _, split_point = fir.load_and_split_data(participant, condition)
    _, emb_test = fir.load_and_split_embeddings(embedding_path, split_point)
    
    weights = read_pickle(f"{filepath}/{participant}/{condition}/model_weights.pkl")
    predicted_signal = np.dot(emb_test, weights)
    del weights
    gc.collect()
    
    # run permutation testing
    correlations,pvals = permutation_test(fmri_test, predicted_signal, nperms=npermutations)

    reject, qvals, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), alpha=fdr_alpha, method='fdr_bh')
    sig_voxel_idx = np.where(reject == 1)[0]

    # save voxel indices, load result data structure and save entries
    print(f"Participant {participant} {condition}: {reject.sum()}/{len(reject)} voxels significant at FDR {fdr_alpha}. {np.sum(pvals < 0.01)} voxels significant at p < 0.01. {np.sum(pvals < 0.05)} at p < 0.05.")
    del fmri_test, emb_test, predicted_signal, correlations, pvals #, qvals, reject
    gc.collect()

    # delete variables and gc collect
    return sig_voxel_idx


def iterate_through_participants(filepath):
    participants = os.listdir(filepath)
    
    records = []
    # iterating through participants
    
    for p in participants:
        print(p)
        for c in conditions:
            datapath = f"{filepath}/{p}/{c}"
            stat_file = f"{datapath}/correlations.pkl"
            
            if not os.path.isfile(stat_file):
                print(f"No file for Participant {p}. Skipping.")
                continue
            
            sig_voxel_idx = permutation_test_wrapper(filepath, p,c)

            with open(stat_file, 'rb') as f:
                try:
                    stat_vec = pickle.load(f) # stat vec is just voxels from the schaefer parcel, about 130k voxels
                except:
                    print(f"issue with {p}: {stat_file}")

            # filter to top 10k, 10k is default parameter but can be changed with threshold argument
            z_vec = np.arctanh(stat_vec)
            cutoff = find_cutoff(z_vec)

            top_voxel_idx = (np.where(z_vec > cutoff))[0]

            top_voxel_vals = z_vec[top_voxel_idx]
            # Using z-scored correlation coefficients for downstream correlation tests
            top_vals = z_vec[np.where(z_vec > cutoff)[0]]
            participant_mean = float(np.mean(top_vals))
        
            # parcel nums contains schaefer parcel numbers for each voxel location
            # so this finds the corresponding parcels for the top voxels
            top_parcels = map_to_schaefer_regions(top_voxel_idx)
            sig_parcels = map_to_schaefer_regions(sig_voxel_idx)
            
            # Mapping directly from voxel indices to HarvOx regions
            top_regions = map_to_HarvOx_regions(top_voxel_idx)
            sig_regions = map_to_HarvOx_regions(sig_voxel_idx)

            # True/False per entry of top_voxel_idx: whether that voxel is covered by HarvOx (False = ignore for HarvOx comparisons)
            top_voxel_in_hox_mask = in_harvox_mask(top_voxel_idx)

            new_record = {
                'participant' : p,
                'condition' : c,
                'top_voxel_idx' : top_voxel_idx,
                'top_voxel_vals' : top_voxel_vals,
                'top_voxel_in_hox_mask' : top_voxel_in_hox_mask,
                'participant_mean' : participant_mean,
                'top_parcels' : top_parcels,
                'top_regions' : top_regions,
                'sig_voxel_idx': sig_voxel_idx,
                'sig_parcels' : sig_parcels,
                'sig_regions' : sig_regions
            }
            records.append(new_record)
        #     break
        # break

    records = pd.DataFrame(records)
    return records

def main():
    # iterate through directories, parse the file names, and accumulate stats
    filepath = "output/VEMs"
    outpath = f"{filepath.removesuffix('/VEMs')}/organized_results_by_condition.pkl"
    
    # participant_means, stat_collection, parcel_collection 
    records = iterate_through_participants(filepath)
    write_pickle(records, outpath)

if __name__ == "__main__":
    main()
        