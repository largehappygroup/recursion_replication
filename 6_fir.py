import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import gc
import glob
import math
import pickle
import warnings
import traceback
import numpy as np
import nibabel as nib
from npp import zscore
import matplotlib.pyplot as plt
from ridge import bootstrap_ridge
from matplotlib.pyplot import figure, cm
from concurrent.futures import ProcessPoolExecutor, as_completed

ALLOWED_CORES = list(range(0,30))

CONDITIONS = ['recursion', 'iteration', 'prose', 'list']

#############################################################################
############## Loading Atlases ##############################################
#############################################################################%


# read in 2d mni mask
mask = nib.load(f"mask/MNI152_T1_2mm_brain_mask.nii.gz")
og_shape = mask.shape
mask = mask.get_fdata().flatten()
brain_idx = np.where(mask>0)[0]

atlas = nib.load(f"misc/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
atlas_vec = atlas.get_fdata().flatten()
atlas_only_brain = atlas_vec[brain_idx]
cortex = np.where(atlas_only_brain != 0)[0]

# Making empty templates to save output
empty_schaefer = np.zeros(atlas_only_brain.shape)
empty_mni = np.zeros(atlas_vec.shape)

#############################################################################
############## Functions ####################################################
#############################################################################
    
# For plotting purposes, finding the voxels with the highest performance
def find_top_voxels(corrs):
    max_corrs = corrs.copy()
    max_corrs.sort()
    max_corrs = max_corrs[-5:]
    top_inds = [int(np.where(corrs == c)[0][0]) for c in max_corrs]
    
    return top_inds
    
# Making plots for correlations and predicted vs. actual signal
def make_and_save_plots(outpath, bscorrs, fmri_test, stat, predicted_signal, emb_test, keys_test, stat_used):
    
    warnings.filterwarnings("ignore", message="No artists with labels found")
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")
    
    # Plotting training performance
    f = figure()
    ax = f.add_subplot(1,1,1)
    ax.semilogx( np.logspace(1,3,12), bscorrs.mean(2).mean(1), 'o-')
    plt.savefig(f"{outpath}/training_performance.png", dpi=150)
    
    # Plotting top 5 timecourses along with keystrokes
    top_voxels = find_top_voxels(stat)
    f = figure(figsize=(15,15))
    
    for i in range(1,6):
        
        ax = f.add_subplot(5,1,i)
        selvox = top_voxels[i-1]

        realresp = ax.plot(fmri_test[:,selvox], 'k')[0]
        predresp = ax.plot(zscore(predicted_signal[:,selvox]), 'r')[0]
        ax.set_ylabel(f"Voxel {selvox}")

        ax.set_xlim(0, len(keys_test))
        ax.set_xlabel(f"Time (fMRI time points)")
        ax.legend()
        if i == 5:
            x_labels = list(keys_test.values())
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)

        ax.legend((realresp, predresp), ("Actual response", "Predicted response (scaled)"));
    plt.savefig(f"{outpath}/top_5_voxels.png", dpi=150)
    plt.close('all')
    gc.collect()

# emb_train, fmri_train, emb_test, fmri_test
def run_ridge_regression(emb_train, fmri_train, emb_test, fmri_test):
    alphas = np.logspace(1, 3, 12) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 5 #1 # Number of cross-validation runs.
    chunklen = 40 #
    '''
    To select the ridge parameter independently for each voxel, we used 50 iterations
    of cross-validation. Since fMRI data is auto-correlated, for
    each cross-validation run we randomly sampled 40 different
    chunks of the training data, each totaling over 4 minutes.
    The training set comprised 26 stories, totaling 5.4 hours.
    '''
    # nchunks=20 (holding out 800 TRs) assumes a training set of thousands of TRs, as in
    # the combined, all-conditions matrix this was originally tuned on. Per-condition
    # training sets can be much smaller (e.g. 'list' is consistently the smallest
    # condition), so nchunks is scaled to keep the held-out set at ~20% of the training
    # data -- otherwise bootstrap_ridge can hold out the entire training set, leaving
    # nothing to fit on and crashing in ridge_corr's SVD.
    n_train = emb_train.shape[0]
    nchunks = max(1, min(20, math.floor(0.2 * n_train / chunklen)))
    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(emb_train, fmri_train, emb_test, fmri_test,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)

    return wt, corr, bscorrs

def calculate_R2(predicted_signal, actual_signal):
    SS_res = np.sum((actual_signal - predicted_signal) ** 2, axis=0)
    SS_tot = np.sum((actual_signal - actual_signal.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - (SS_res / SS_tot)
    return R2
    
def ridge_regression_wrapper(embedding_path, participant, bass_outpath, split_point, fmri_train, fmri_test, keys_test):
    
    corr_outfile = f"{bass_outpath}/correlations.pkl"
    weights_outfile = f"{bass_outpath}/model_weights.pkl"
    keys_outfile = f"{bass_outpath}/test_keystrokes.pkl"
    R2_outfile = f"{bass_outpath}/R2.pkl"

    if os.path.isfile(corr_outfile) and os.path.isfile(keys_outfile) and os.path.isfile(R2_outfile) and os.path.isfile(weights_outfile):
        return

    emb_train, emb_test = load_and_split_embeddings(embedding_path, split_point)

    # Running ridge regression here
    print(f"Participant {participant} Ridge Regression")
    weights,corrs,bscorrs = run_ridge_regression(emb_train, fmri_train, emb_test, fmri_test)
    predicted_signal = np.dot(emb_test, weights)
    R2 = calculate_R2(predicted_signal, fmri_test)
    
    # Saving model weights and correlation values between predicted and actual timecourses as output
    if not os.path.exists(bass_outpath):
        os.makedirs(bass_outpath)
    
    with open(weights_outfile, 'wb') as f:
        pickle.dump(weights, f)

    with open(corr_outfile, 'wb') as f:
        pickle.dump(corrs, f)
        
    with open(keys_outfile, 'wb') as f:
        pickle.dump(keys_test, f)
        
    with open(R2_outfile, 'wb') as f:
        pickle.dump(R2, f)
        
    make_and_save_plots(bass_outpath, bscorrs, fmri_test, corrs, predicted_signal, emb_test, keys_test, 'correlation')
    
    # Trying to free up space
    del weights, corrs, bscorrs, emb_train, emb_test, fmri_train, fmri_test
    gc.collect()
    return
    
def load_and_split_embeddings(embedding_path, split_point):
    
    with open(embedding_path, 'rb') as f:
        embedding = pickle.load(f)

    delRstim = embedding[:split_point, :] # delRstim from pickle files
    delPstim = embedding[split_point:, :] # delPstim is prediction
    
    return delRstim, delPstim

def split_keystrokes(all_keystrokes, split_point):
    Rkeys = {i:v for i,v in enumerate(all_keystrokes) if i < split_point}
    Pkeys = {i:v for i,v in enumerate(all_keystrokes) if i >= split_point}
    return Rkeys,Pkeys

def fmri_train_test_split(reshaped_scan):
    
    split_point = math.floor((reshaped_scan.shape)[0]*0.9)
    zRresp = reshaped_scan[:split_point,:] # zRresp is fMRI data for training
    zPresp = reshaped_scan[split_point:, :] # zPresp is fMRI data for prediction
    
    return zRresp, zPresp, split_point
    
def load_and_reshape_fmri_data(fmripath, vols_to_skip):
    fmri_data = nib.load(fmripath)
    scan = fmri_data.get_fdata()
    scan_2d = (np.reshape(scan, [scan.shape[0]*scan.shape[1]*scan.shape[2], scan.shape[3]]))

    means = scan_2d.mean(axis=1, keepdims=True)
    stds = scan_2d.std(axis=1, keepdims=True)

    z_scored_2d_scan = (scan_2d - means) / np.where(stds==0, 1, stds)

    # Only looking at voxels in the MNI brain
    scan_2d_brain = z_scored_2d_scan[brain_idx, :]

    # Only looking at voxels labeled in the Schaefer Atlas
    scan_2d_schaefer = scan_2d_brain[cortex,:]
    scan_2d_schaefer = scan_2d_schaefer.T
    
    vol_nums = np.ones((scan_2d_schaefer.shape)[0])

    # TODO - skip scans that got cut short during preprocessing
    # vols_to_skip is empty when a scan block is entirely one condition (nothing to
    # exclude for it), so guard against max() on an empty sequence
    if vols_to_skip and max(vols_to_skip) > scan_2d_schaefer.shape[0]:
        print(f"Scan got clipped, skipping.")
        return [], []

    vol_nums[vols_to_skip] = 0
    scan_2d_schaefer_filtered = scan_2d_schaefer[np.where(vol_nums == 1)]
    
    return scan_2d_schaefer_filtered, vol_nums

def isolate_scan_num(f):
    file = f.split('/')[-1]
    scan = file.removesuffix('.nii.gz')
    return scan

def load_and_split_data(p, condition):

    participant_fmri_paths = glob.glob(f"clean_data/{p}/*.nii.gz")
    # print(participant_fmri_paths)
    participant_fmri_paths = sorted(participant_fmri_paths, key=lambda f: isolate_scan_num(f))

    all_scans = []
    all_keystrokes = []
    for fmri_file in participant_fmri_paths:
        scan_num = isolate_scan_num(fmri_file)
        keystroke_path = glob.glob(f"midprocess/{p}/scan_{scan_num}*-new_keystrokes.pkl")[0]
        # condition-specific mask, written by 5_prepare_embeddings_for_fir.py, so the
        # fMRI volumes kept here line up row-for-row with this condition's stimulus matrix
        vols_to_skip_path = glob.glob(f"midprocess/{p}/scan_{scan_num}*-{condition}-vols_to_skip.pkl")[0]

        with open(vols_to_skip_path, 'rb') as f:
            vols_to_skip = pickle.load(f)

        with open(keystroke_path, 'rb') as f:
            keystrokes = pickle.load(f)

        atlas_voxels_2d,vol_nums = load_and_reshape_fmri_data(fmri_file, vols_to_skip)
        if len(atlas_voxels_2d) == 0 and len(vol_nums) == 0: # some scans that got cut short during preprocessing :/ Working to fix
            continue

        filtered_keystrokes = {kv[0]:kv[1] for i,kv in enumerate(keystrokes.items()) if vol_nums[i] == 1}

        all_scans.append(atlas_voxels_2d)
        all_keystrokes.extend(list(filtered_keystrokes.values()))

    all_scans = np.vstack(all_scans)

    fmri_train,fmri_test,split_point = fmri_train_test_split(all_scans)
    keys_train,keys_test = split_keystrokes(all_keystrokes, split_point)

    return fmri_train, fmri_test, keys_train, keys_test, split_point

def process_participant(p, condition):
    bass_outpath = f"output/VEMs/{p}/{condition}"
    embedding_path = f"output/stimulus_matrices/{p}-{condition}-stimulus_matrix.pkl"

    try:
        fmri_train, fmri_test, keys_train, keys_test, split_point = load_and_split_data(p, condition)
    except:
        print(f"Issue with data for Participant {p} on {condition}")
        return

    ridge_regression_wrapper(embedding_path, p, bass_outpath, split_point, fmri_train, fmri_test, keys_test)



def init_worker():
    os.sched_setaffinity(0, ALLOWED_CORES)

def list_participant_condition_jobs(stimulus_matrix_dir):
    """Each per-condition stimulus matrix is named
    {participant}-{condition}-stimulus_matrix.pkl; not every participant has a matrix for
    every condition (e.g. a condition can be missing entirely if some of their scan
    blocks are incomplete), so the job list is built from whatever matrices actually
    exist rather than assuming the full participant x condition grid. This also skips
    over the older, pre-condition-split {participant}-stimulus_matrix.pkl files still
    sitting in this directory."""
    jobs = []
    for f in os.listdir(stimulus_matrix_dir):
        if not f.endswith('-stimulus_matrix.pkl'):
            continue
        stem = f.removesuffix('-stimulus_matrix.pkl')
        if '-' not in stem:
            continue
        p, condition = stem.rsplit('-', 1)
        if condition not in CONDITIONS:
            continue
        jobs.append((p, condition))
    return jobs

def main():
    participant_path = "output/stimulus_matrices"
    jobs = list_participant_condition_jobs(participant_path)
    # print(jobs)

    # problematic_files
    # 130 - new keystrokes - looks like there were technical issues with scan-501 where no task data got recorded for the first block
    # 130 - skip the scans with a, also skip 501 due to technical difficulties
    # jobs = [(p, c) for p, c in jobs if p in ('107')]
    # print(jobs)

    num_jobs = len(jobs)
    print(f"{num_jobs} jobs across {len(ALLOWED_CORES)} workers")
    with ProcessPoolExecutor(max_workers=len(ALLOWED_CORES), initializer=init_worker) as ex:
        futures = {
            ex.submit(process_participant, p, condition) : (p, condition) for p, condition in jobs
        }
        for i,fut in enumerate(as_completed(futures), start=1):
            p, condition = futures[fut]
            try:
                fut.result()
                print(f"  [{i}/{num_jobs}] done: {p} ({condition})")
            except Exception:
                print(f"  [{i}/{num_jobs}] error: {p} ({condition})")
                traceback.print_exc()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()