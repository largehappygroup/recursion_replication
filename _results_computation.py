import pickle
import cortex
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

N_PARCELS = 401  # 400 schaefer parcels + 0 for non-cortical voxels


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def permutation_pval(obs, randos, nperms):
    randos = np.array(randos)
    greater = np.sum(randos >= obs)
    lesser = np.sum(randos <= obs)
    tail = min(greater, lesser)
    pval = 2 * (1 + tail) / (1 + nperms)
    return min(pval, 1.0)

def set_diff(arr_a, arr_b):
    mask = ~np.isin(arr_a, arr_b)   # True where arr_a's elements are NOT in arr_b
    diff_vals = arr_a[mask]
    diff_idx = np.nonzero(mask)[0]
    return diff_vals,diff_idx

def get_top_regions(parcel_dict, nregions=10):
    parcel_dict = dict(sorted(parcel_dict.items(), key= lambda x: x[1], reverse=True))
    return {region : val for i,(region,val) in enumerate(parcel_dict.items()) if i < nregions}

def convert_count_to_proportion(count_dict, parcel_voxel_counts):
    new_dict = {}
    for parcel_num, count in count_dict.items():
        proportion = count / parcel_voxel_counts[parcel_num]
        new_dict[parcel_num] = proportion
    return new_dict

def increment_parcels(top_parcels, schaefers, subset):
    for p in top_parcels:
        schaefers[subset][p] += 1
    return schaefers

def organize_and_count_parcels(regions, idx, subset, schaefers):
    subset_of_regions = regions[idx]
    counts = dict(Counter(subset_of_regions))
    # proportions = convert_count_to_proportion(counts, parcel_voxel_counts)
    # top_proportions = get_top_regions(proportions)
    top_proportions = get_top_regions(counts)
    increment_parcels(top_proportions, schaefers, subset)

    return top_proportions


def compare_conditions(records, nulls, cond_a, cond_b, control_a=None, control_b=None,
                       participants=None, exclude=(), nperms=5000):
    """
    For every participant, count the top voxels shared between cond_a and cond_b,
    test that count against the participant's null distribution, and tally the
    top-10 schaefer parcels of the shared and condition-specific voxels.

    Single contrast (no controls given), e.g. recursion vs iteration:
        cond_a-specific = cond_a minus cond_b
        cond_b-specific = cond_b minus cond_a
        shared          = cond_a intersect cond_b
    Double contrast (control_a and control_b given), e.g. (recursion - iteration) vs (prose - list):
        cond_a-specific = cond_a minus control_a
        cond_b-specific = cond_b minus control_b
        shared          = cond_a-specific intersect cond_b-specific

    participants: only include these participant ids (None = everyone)
    exclude:      skip these participant ids
    """
    if (control_a is None) != (control_b is None):
        raise ValueError("control_a and control_b must be given together")
    double_contrast = control_a is not None

    # keys into the null distribution pickle (see _null_distributions.py)
    if double_contrast:
        null_key = (f'{cond_a}-{control_a}', f'{cond_b}-{control_b}')
        key_by_subset = {
            cond_a  : (cond_a, control_a, f'{cond_a}_unique'),
            cond_b  : (cond_b, control_b, f'{cond_b}_unique'),
            'shared': (*null_key, 'shared'),
        }
        required_conditions = {cond_a, cond_b, control_a, control_b}
    else:
        null_key = (cond_a, cond_b)
        key_by_subset = {
            cond_a  : (cond_a, cond_b, f'{cond_a}_unique'),
            cond_b  : (cond_a, cond_b, f'{cond_b}_unique'),
            'shared': (cond_a, cond_b, 'shared'),
        }
        required_conditions = {cond_a, cond_b}

    participant_ids = []
    observed_vals = []
    null_vals_list = []
    pvals = []
    schaefers = {subset: np.zeros(N_PARCELS) for subset in key_by_subset}

    for pid,mini_df in records.groupby(['participant']):
        pid = pid[0]

        if pid in exclude:
            continue
        if participants is not None and pid not in participants:
            continue

        top_voxels = mini_df.set_index('condition')['top_voxel_idx']
        if not required_conditions.issubset(top_voxels.keys()):
            continue
        top_parcels = mini_df.set_index('condition')['top_parcels']
        all_a_parcels = np.array(top_parcels[cond_a])
        all_b_parcels = np.array(top_parcels[cond_b])

        if double_contrast:
            # find voxels unique to each condition relative to its control
            _,a_unique_idx = set_diff(top_voxels[cond_a], top_voxels[control_a])
            _,b_unique_idx = set_diff(top_voxels[cond_b], top_voxels[control_b])

            a_unique_vox = top_voxels[cond_a][a_unique_idx]
            b_unique_vox = top_voxels[cond_b][b_unique_idx]

            shared_voxels, shared_idx_within_unique,_ = np.intersect1d(a_unique_vox, b_unique_vox, return_indices=True)
            # shared_idx_within_unique is a position within a_unique_vox (a filtered subset), not
            # the full cond_a array - compose back through a_unique_idx before indexing all_a_parcels
            shared_idx = a_unique_idx[shared_idx_within_unique]
        else:
            # find voxels where the two conditions intersect
            shared_voxels, shared_idx,_ = np.intersect1d(top_voxels[cond_a], top_voxels[cond_b], return_indices=True)

            # find voxels unique to the two conditions
            _,a_unique_idx = set_diff(top_voxels[cond_a], top_voxels[cond_b])
            _,b_unique_idx = set_diff(top_voxels[cond_b], top_voxels[cond_a])
        num_shared = len(shared_voxels)

        # Brain locations
        organize_and_count_parcels(all_a_parcels, shared_idx, 'shared', schaefers)
        organize_and_count_parcels(all_a_parcels, a_unique_idx, cond_a, schaefers)
        organize_and_count_parcels(all_b_parcels, b_unique_idx, cond_b, schaefers)

        # Load in null distribution values of intersection
        pair_nulls = nulls[pid][null_key]
        pval = permutation_pval(num_shared, pair_nulls, nperms)

        # For plotting purposes
        participant_ids.append(pid)
        observed_vals.append(num_shared)
        null_vals_list.append(np.array(pair_nulls))
        pvals.append(pval)

    return {
        'participant_ids': participant_ids,
        'observed_vals'  : observed_vals,
        'null_vals_list' : null_vals_list,
        'pvals'          : pvals,
        'schaefers'      : schaefers,
        'key_by_subset'  : key_by_subset,
    }


def plot_shared_voxels(participant_ids, observed_vals, null_vals_list, title):
    n = len(participant_ids)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.6), 6))

    parts = ax.violinplot(null_vals_list, positions=range(n), widths=0.7,
                           showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('none')
        pc.set_alpha(0.4)
        pc.set_zorder(1)

    ax.scatter(range(n), observed_vals, color='#fc996a', s=70, edgecolor='black',
               linewidth=0.6, zorder=2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(participant_ids, rotation=90)
    ax.set_xlabel('Participant')
    ax.set_ylabel('Number of Top Voxels Shared Between Conditions')
    ax.set_title(title)

    legend_elements = [
        Patch(facecolor='gray', alpha=0.4, label='Null distribution'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='darkorange',
               markeredgecolor='black', markersize=8, label='Observed intersection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Group-level significance for the schaefer parcel tallies: for each parcel,
# is the number of participants who have it in their top-10 higher than
# chance? Chance is built by holding each participant's real top-voxel /
# shared / unique sets fixed and permuting only the voxel -> parcel label
# mapping (see _null_distributions.permute_top_parcels_for_pair /
# permute_double_contrast_top_parcels), then summing each participant's
# permuted top-10 indicator across participants to get a null distribution
# for the group tally.
def build_group_null(nulls, participant_ids, key):
    contributing = [nulls[pid][key] for pid in participant_ids if key in nulls.get(pid, {})]
    return np.sum(contributing, axis=0)  # (nperms, n_parcels)

def parcel_pvals(observed, group_null):
    nperms = group_null.shape[0]
    # one-sided: testing whether the observed tally is higher than chance
    return (1 + np.sum(group_null >= observed, axis=0)) / (1 + nperms)

def threshold_schaefers(schaefers, nulls, participant_ids, key_by_subset, alpha):
    thresholded = {}
    schaefer_qvals = {}
    for subset, observed in schaefers.items():
        group_null = build_group_null(nulls, participant_ids, key_by_subset[subset])
        pvals = parcel_pvals(observed, group_null)

        reject, qvals, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), alpha=0.05, method='fdr_bh')

        schaefer_qvals[subset] = qvals
        thresholded[subset] = np.where(qvals < alpha, observed, 0)
    return thresholded, schaefer_qvals


def convert_to_nifti(values, atlases):
    # working backwards to save correlation values as voxels in MNI space
    new_schaefer = atlases.empty_schaefer.copy()
    new_mni = atlases.empty_mni.copy()

    new_schaefer[atlases.cortex_vx] = values
    new_mni[atlases.brain_idx] = new_schaefer
    result_brain = np.reshape(new_mni, atlases.og_shape)

    # Saving results
    nifti_result = nib.Nifti1Image(result_brain, affine=atlases.atlas.affine, header=atlases.atlas.header)
    # nib.save(nifti_result, "test_plotting.nii.gz")
    return result_brain, nifti_result

def find_max(schaefers):
    count_max = 0
    for subset,counts in schaefers.items():
        if max(counts) > count_max:
            count_max = max(counts)
    return count_max

def save_schaefer_maps(schaefers, atlases, vmax):
    # find indices of voxels corresponding to each schaefer parcel
    for group, lookup in schaefers.items():
        print(group)
        result = lookup[atlases.schaefer_voxels.astype(int)]

        npy_brain, nifti_brain = convert_to_nifti(result, atlases)
        npy_brain = npy_brain.transpose(2,1,0)

        vol = cortex.Volume(
                npy_brain,
                subject='fsaverage',
                xfmname='mni2py2',
                vmin=0,
                vmax=vmax,
                description=f"{group}",
                cmap='magma_r',
        )
        cortex.webshow(vol, overlays_visible=('sulci'),labels_visible=())
