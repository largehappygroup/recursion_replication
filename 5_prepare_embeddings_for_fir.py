import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import re
import glob
import pickle
import traceback
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

ALLOWED_CORES = list(range(0, 30))

# same 4 conditions/column order as make_design_matrices.py
CONDITIONS = ['recursion', 'iteration', 'prose', 'list']

with open("misc/outlier_volumes.pkl", 'rb') as f:
    outlier_vols = pickle.load(f)

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        # print(f"iteration {d}", dstim.shape)
        
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def reduce_dimensionality(X):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.99) # adjusted to 99% variance on 2/16/2026
    try:
        X_reduced = pca.fit_transform(X_scaled)
    except Exception as e:
        print("Issue with standard PCA, selecting fixed number of components")
        n_components = 512 # Based on the amount of explained variance when testing PCA components
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        with open("preparing_embedding_error_log.txt", '+a') as f:
            f.write(f"Issue with PCA: {e}\n")
    return X_reduced

def prepare_regressor(participant, scan_block, vols_to_skip):
    num_keys_regressor_path = f"midprocess/{participant}/{scan_block}-num_keystrokes_regressor.pkl"
        
    with open(num_keys_regressor_path, 'rb') as f:
        num_keys_regressor = pickle.load(f)
    
    mean = np.mean(num_keys_regressor) 
    std = np.std(num_keys_regressor)

    num_keys_regressor = [(n - mean)/std for n in num_keys_regressor]

    num_keys_regressor = [n for i,n in enumerate(num_keys_regressor) if i not in vols_to_skip]
    num_keys_regressor = np.expand_dims(np.array(num_keys_regressor), axis=1)
    
    return num_keys_regressor

def determine_column(stim_id):
    """Maps a stim_id to its condition index [recursion 0, iteration 1, prose 2, list 3].
    Must stay consistent with make_design_matrices.py's determine_column."""
    if stim_id < 200:
        return 0
    elif stim_id < 300:
        return 1
    elif stim_id < 400:
        return 2
    else:
        return 3

def load_keystroke_df(participant, scan_block):
    path = f"midprocess/{participant}/{scan_block}_keystroke_df.csv"
    return pd.read_csv(path)

def assign_volume_conditions(keystroke_df):
    """Condition index per volume: the active question's condition, or, during rest,
    the condition of the question that just finished (carried forward to account for
    the hemodynamic delay). The rest before the block's first question has no
    preceding question, so it's attributed to that first question instead -- this
    keeps those initial volumes in (rather than dropping them), separating this
    block from whatever precedes it and giving the BOLD signal room to return to
    baseline before the block's own first question starts."""
    conditions = []
    current = None
    for stim_id in keystroke_df['stim_id']:
        if pd.notna(stim_id):
            current = determine_column(int(stim_id))
        conditions.append(current)

    first_known = next((c for c in conditions if c is not None), None)
    conditions = [c if c is not None else first_known for c in conditions]
    return conditions

def organize_scan_block(participant, scan_block, keystroke_dict, embedding_dict, keystroke_df):
    """Builds per-volume records for one scan block, in volume order, skipping outlier
    volumes (condition is None only if the block has no questions at all). Every rest
    volume -- leading or trailing -- is kept as a record with no embedding (filled with
    zeros once the signal is assembled), so it stays in its associated question's
    condition matrix without contributing a new stimulus.

    Returns records (list of (vol, condition, is_question, raw_embedding_or_None)) and
    conditions (per-volume condition index or None, same length as keystroke_df).
    """
    num_vols = len(keystroke_df)

    # indices of keystroke_dict are 0-indexed
    # check to see if participant has outlier volumes {'participant': {'scan_block' : [vols]}}
    outlier_set = set()
    if participant in list(outlier_vols.keys()) and scan_block in list(outlier_vols[participant].keys()):
        outlier_set.update(int(n) for n in outlier_vols[participant][scan_block])

    conditions = assign_volume_conditions(keystroke_df)

    records = []
    for vol in range(num_vols):
        condition = conditions[vol]
        if condition is None or vol in outlier_set:
            continue

        if vol in keystroke_dict:
            records.append((vol, condition, True, embedding_dict[keystroke_dict[vol]]))
        else:
            records.append((vol, condition, False, None))

    return records, conditions

def isolate_scan_num(filename):
    file = filename.split('/')[-1]
    scan_match = re.match(r'^scan_(\d+)-*', file)
    scan = scan_match.group(1)
    return scan

def process_participant_lookahead(p, embedding_path, ndelays):

    emb_bass_datapath = f"{embedding_path}/{p}"
    emb_files = glob.glob(f"{emb_bass_datapath}/*-keystroke_embeddings.pkl")

    # sort by the scan numbers in embedding files
    emb_files = sorted(emb_files, key=lambda f: isolate_scan_num(f))

    # iterating through the different scans and concatenating into one, in volume order
    # each record is (condition, is_question, raw_embedding_or_None)
    all_records = []
    all_regressors = []
    for file in emb_files:
        try:
            with open(file, 'rb') as f:
                embedding_dict = pickle.load(f)
        except Exception as e:
            print(f"can't open embedding for {p}: {e}")
            return

        scan_block_file = file.split('/')[-1]
        scan_block = scan_block_file.removesuffix('-keystroke_embeddings.pkl')
        keystroke_df = load_keystroke_df(p, scan_block)
        keystroke_path = f"midprocess/{p}/{scan_block}_keystroke_log.pkl"

        try:
            with open(keystroke_path, 'rb') as f:
                keystroke_dict = pickle.load(f)
        except Exception as e:
            print(f"can't open keystrokes for {p}: {e}")
            return

        records, conditions = organize_scan_block(str(p), scan_block, keystroke_dict, embedding_dict, keystroke_df)
        included_vols = {vol for vol, *_ in records}
        excluded_vols = sorted(set(range(len(keystroke_df))) - included_vols)

        regressor = prepare_regressor(p, scan_block, excluded_vols)

        for (_, condition, is_question, raw_embedding), reg_val in zip(records, regressor):
            all_records.append((condition, is_question, raw_embedding))
            all_regressors.append(reg_val)

        # per-condition exclusion masks, for matching this block's fMRI volumes up
        # against each condition's stimulus matrix later
        for cond_idx, cond_name in enumerate(CONDITIONS):
            cond_excluded_vols = sorted(
                v for v in range(len(keystroke_df))
                if conditions[v] != cond_idx or v not in included_vols
            )
            with open(f"midprocess/{p}/{scan_block}-{cond_name}-vols_to_skip.pkl", 'wb') as f:
                pickle.dump(cond_excluded_vols, f)

    # fit dimensionality reduction once, across every condition's real (non-rest) signal
    question_embeddings = np.vstack([r[2] for r in all_records if r[1]])
    sig_pca = reduce_dimensionality(question_embeddings)

    # rebuild the full per-volume sequence: real rows get their PCA-reduced embedding,
    # rest rows are left as zeros (no new stimulus, but kept in for temporal contiguity
    # around the FIR delays -- see assign_volume_conditions)
    full_embeddings = np.zeros((len(all_records), sig_pca.shape[1]))
    q_i = 0
    for i, (_, is_question, _) in enumerate(all_records):
        if is_question:
            full_embeddings[i] = sig_pca[q_i]
            q_i += 1

    all_regressors = np.vstack(all_regressors)
    sig_with_regressor = np.hstack((all_regressors, full_embeddings))

    conditions_arr = np.array([r[0] for r in all_records])
    delays = range(1, ndelays + 1)
    for cond_idx, cond_name in enumerate(CONDITIONS):
        cond_sig = sig_with_regressor[conditions_arr == cond_idx]
        if len(cond_sig) == 0:
            print(f"Participant {p} has no volumes for condition {cond_name}, skipping.")
            continue

        delayed_sig = make_delayed(cond_sig, delays)
        print(f"Participant {p} {cond_name} signal shape: {delayed_sig.shape}")

        out_file = f"output/stimulus_matrices/{p}-{cond_name}-stimulus_matrix.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(delayed_sig, f)

def init_worker():
    os.sched_setaffinity(0, ALLOWED_CORES)

def run_participants(embedding_path):
    participants = os.listdir(embedding_path)
    ndelays = 10
    
    jobs = [p for p in participants]
    num_jobs = len(jobs)
    print(f"{num_jobs} jobs across {len(ALLOWED_CORES)} workers")
    with ProcessPoolExecutor(max_workers=len(ALLOWED_CORES), initializer=init_worker) as ex:
        futures = {
            ex.submit(process_participant_lookahead, p, embedding_path, ndelays): p
            for p in jobs
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            p = futures[fut]
            try:
                fut.result()
                print(f"  [{i}/{num_jobs}] done: {p}")
            except Exception:
                print(f"  [{i}/{num_jobs}] error: {p}")
                traceback.print_exc()

def main():
    embedding_path = 'output/embeddings'
    run_participants(embedding_path)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
