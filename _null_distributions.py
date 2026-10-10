import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict, Counter

seed = 888
# nperms = 5000
nperms = 10000 # there were some p-values around the significance threshold at 5000, so increasing perms here for increased stability
n_parcels = 401
rng = np.random.default_rng(seed)

def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle(object, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)

# def find_cutoff(vec, threshold = 10**4):
def find_cutoff(vec, threshold = 6525): # average number of significant voxels
    copy = vec.copy()
    copy.sort()
    cutoff = copy[-threshold-1]
    return cutoff

def permute_and_threshold(array):
    permuted = rng.permutation(array)
    cutoff = find_cutoff(permuted)
    # converting to a set, which makes every element an int64, so converting those to regular ints
    top_voxel_idx = set(int(i) for i in np.where(permuted > cutoff)[0])
    return top_voxel_idx

def load_and_filter_files(files):
    data = {}
    for cond,path in files.items():
        if not os.path.isfile(path):
            data[cond] = []
        else:
            data[cond] = read_pickle(files[cond])
            data[cond] = np.arctanh(data[cond])
    return data


def permute_correlations(person, datapath, nperms):

    files = {
        "recursion" : f"{datapath}/{person}/recursion/correlations.pkl",
        "iteration" : f"{datapath}/{person}/iteration/correlations.pkl",
        "prose"     : f"{datapath}/{person}/prose/correlations.pkl",
        "list"      : f"{datapath}/{person}/list/correlations.pkl"
    }

    # filter so that only files that exist are in the dictionary
    data = load_and_filter_files(files)

    # randomize voxel locations for each list of correlations
    null_intersections = defaultdict(list)
    for i in range(nperms):
        permuted = data.copy()
        top_permuted_idx = {condition: permute_and_threshold(correlations) for condition,correlations in permuted.items() if len(correlations) > 0}

        intersections = {
            (cond1,cond2) : len(set.intersection(top_permuted_idx[cond1], top_permuted_idx[cond2]))
            for cond1,cond2 in combinations(top_permuted_idx, 2)
        }

        # updating results
        for k,v in intersections.items():
            null_intersections[k].append(v)

        # double intersection: indices specific to recursion (recursion - iteration)
        # intersected with indices specific to prose (prose - list)
        if {'recursion', 'iteration', 'prose', 'list'}.issubset(top_permuted_idx):
            recursion_specific = top_permuted_idx['recursion'] - top_permuted_idx['iteration']
            prose_specific = top_permuted_idx['prose'] - top_permuted_idx['list']
            double_intersection = recursion_specific & prose_specific
            null_intersections[('recursion-iteration', 'prose-list')].append(len(double_intersection))

            # # double intersection: indices specific to iteration (iteration - recursion)
            # # intersected with indices specific to list (list - prose)
            # iteration_specific = top_permuted_idx['iteration'] - top_permuted_idx['recursion']
            # list_specific = top_permuted_idx['list'] - top_permuted_idx['prose']
            # double_intersection = iteration_specific & list_specific
            # null_intersections[('iteration-recursion', 'list-prose')].append(len(double_intersection))

    return null_intersections

def set_diff_idx(arr_a, arr_b):
    # indices into arr_a of the elements NOT found in arr_b
    mask = ~np.isin(arr_a, arr_b)
    return np.nonzero(mask)[0]

def get_top_regions(parcel_counts, nregions=10):
    parcel_counts = dict(sorted(parcel_counts.items(), key=lambda x: x[1], reverse=True))
    return [region for i, region in enumerate(parcel_counts) if i < nregions]

def top_parcel_indicator(parcel_labels, nregions=10, n_parcels=n_parcels):
    # 1 where a parcel lands in the top-nregions by voxel count, 0 elsewhere
    indicator = np.zeros(n_parcels, dtype=int)
    counts = dict(Counter(parcel_labels))
    for region in get_top_regions(counts, nregions):
        indicator[int(region)] = 1
    return indicator

def permute_top_parcels_for_pair(top_voxels, top_parcels, cond1, cond2, nperms, rng, n_parcels=n_parcels):
    """
    Null distribution for which Schaefer parcels land in a participant's top-10
    by chance. The real top-voxel sets (and which of them are shared vs. unique
    to each condition) are held fixed; only the voxel -> parcel label mapping is
    permuted, so this tests whether the observed clumping into specific parcels
    exceeds what you'd see from a random subset of that participant's own
    top-modeled voxels.
    """
    all_c1_parcels = np.array(top_parcels[cond1])
    all_c2_parcels = np.array(top_parcels[cond2])

    _, shared_idx, _ = np.intersect1d(top_voxels[cond1], top_voxels[cond2], return_indices=True)
    c1_unique_idx = set_diff_idx(top_voxels[cond1], top_voxels[cond2])
    c2_unique_idx = set_diff_idx(top_voxels[cond2], top_voxels[cond1])

    null = {
        'shared': np.zeros((nperms, n_parcels), dtype=int),
        f'{cond1}_unique': np.zeros((nperms, n_parcels), dtype=int),
        f'{cond2}_unique': np.zeros((nperms, n_parcels), dtype=int),
    }

    for i in range(nperms):
        permuted_c1 = rng.permutation(all_c1_parcels)
        permuted_c2 = rng.permutation(all_c2_parcels)

        null['shared'][i] = top_parcel_indicator(permuted_c1[shared_idx], n_parcels=n_parcels)
        null[f'{cond1}_unique'][i] = top_parcel_indicator(permuted_c1[c1_unique_idx], n_parcels=n_parcels)
        null[f'{cond2}_unique'][i] = top_parcel_indicator(permuted_c2[c2_unique_idx], n_parcels=n_parcels)

    return null

def permute_double_contrast_top_parcels(top_voxels, top_parcels, cond_a1, cond_a2, cond_b1, cond_b2, nperms, rng, n_parcels=n_parcels):
    """
    Null distributions for the three-way split between two unique-voxel sets:
    (cond_a1 - cond_a2) vs. (cond_b1 - cond_b2) - e.g. recursion-specific
    (vs. iteration) vs. prose-specific (vs. list):
      'a_only' : (cond_a1-cond_a2) - (cond_b1-cond_b2)
      'shared' : (cond_a1-cond_a2) ∩ (cond_b1-cond_b2)
      'b_only' : (cond_b1-cond_b2) - (cond_a1-cond_a2)
    Real, observed voxel selections are held fixed; parcel labels are drawn
    from cond_a1's array for 'a_only'/'shared' and cond_b1's array for
    'b_only' (matching which condition's top voxels each subset is drawn from),
    then permuted.
    """
    all_a1_parcels = np.array(top_parcels[cond_a1])
    all_b1_parcels = np.array(top_parcels[cond_b1])

    a1_unique_idx = set_diff_idx(top_voxels[cond_a1], top_voxels[cond_a2])
    b1_unique_idx = set_diff_idx(top_voxels[cond_b1], top_voxels[cond_b2])

    a1_unique_vox = top_voxels[cond_a1][a1_unique_idx]
    b1_unique_vox = top_voxels[cond_b1][b1_unique_idx]

    _, shared_a_idx, _ = np.intersect1d(a1_unique_vox, b1_unique_vox, return_indices=True)
    a_only_idx = set_diff_idx(a1_unique_vox, b1_unique_vox)
    b_only_idx = set_diff_idx(b1_unique_vox, a1_unique_vox)

    # compose back into direct indices into the full cond_a1 / cond_b1 arrays
    shared_a1_idx = a1_unique_idx[shared_a_idx]
    a_only_a1_idx = a1_unique_idx[a_only_idx]
    b_only_b1_idx = b1_unique_idx[b_only_idx]

    null = {
        'a_only': np.zeros((nperms, n_parcels), dtype=int),
        'shared': np.zeros((nperms, n_parcels), dtype=int),
        'b_only': np.zeros((nperms, n_parcels), dtype=int),
    }

    for i in range(nperms):
        permuted_a1 = rng.permutation(all_a1_parcels)
        permuted_b1 = rng.permutation(all_b1_parcels)

        null['a_only'][i] = top_parcel_indicator(permuted_a1[a_only_a1_idx], n_parcels=n_parcels)
        null['shared'][i] = top_parcel_indicator(permuted_a1[shared_a1_idx], n_parcels=n_parcels)
        null['b_only'][i] = top_parcel_indicator(permuted_b1[b_only_b1_idx], n_parcels=n_parcels)

    return null

def add_top_parcel_nulls(null_intersections, records, nperms, rng, n_parcels=n_parcels):
    for pid, mini_df in records.groupby('participant'):
        print(pid)
        top_voxels = mini_df.set_index('condition')['top_voxel_idx']
        top_parcels = mini_df.set_index('condition')['top_parcels']
        present = set(top_voxels.index)

        if {'recursion', 'iteration'}.issubset(present):
            pair_null = permute_top_parcels_for_pair(
                top_voxels, top_parcels, 'recursion', 'iteration', nperms, rng, n_parcels
            )
            for subset, arr in pair_null.items():
                null_intersections[pid][('recursion', 'iteration', subset)] = arr

        if {'prose', 'list'}.issubset(present):
            pair_null = permute_top_parcels_for_pair(
                top_voxels, top_parcels, 'prose', 'list', nperms, rng, n_parcels
            )
            for subset, arr in pair_null.items():
                null_intersections[pid][('prose', 'list', subset)] = arr

        if {'recursion', 'iteration', 'prose', 'list'}.issubset(present):
            double_null = permute_double_contrast_top_parcels(
                top_voxels, top_parcels, 'recursion', 'iteration', 'prose', 'list', nperms, rng, n_parcels
            )
            for subset, arr in double_null.items():
                null_intersections[pid][('recursion-iteration', 'prose-list', subset)] = arr

            # double_null = permute_double_contrast_top_parcels(
            #     top_voxels, top_parcels, 'iteration', 'recursion', 'list', 'prose', nperms, rng, n_parcels
            # )
            # for subset, arr in double_null.items():
            #     null_intersections[pid][('iteration-recursion', 'list-prose', subset)] = arr

    return null_intersections

def main():
    datapath = f"output/VEMs"
    recordspath = "output/organized_results_by_condition.pkl"
    outpath = "output/null_distributions.pkl"
    participants = os.listdir(datapath)

    null_intersections = defaultdict(dict)
    for p in participants:
        print(p)
        intersections = permute_correlations(p, datapath, nperms)
        null_intersections[p] = intersections

    records = read_pickle(recordspath)
    null_intersections = add_top_parcel_nulls(null_intersections, records, nperms, rng, n_parcels)

    # print(null_intersections)
    write_pickle(null_intersections, outpath)


if __name__ == "__main__":
    main()
