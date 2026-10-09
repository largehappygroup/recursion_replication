from dataclasses import dataclass

import numpy as np
import nibabel as nib


@dataclass
class BrainAtlases:
    og_shape: tuple            # 3d shape of the MNI volume
    brain_idx: np.ndarray      # indices of in-brain voxels in the flattened MNI volume
    atlas: nib.Nifti1Image     # schaefer atlas image (affine/header used when saving niftis)
    atlas_only_brain: np.ndarray  # schaefer parcel numbers for every in-brain voxel
    cortex_vx: np.ndarray      # indices (within brain_idx) of voxels that fall in a schaefer parcel
    schaefer_voxels: np.ndarray   # schaefer parcel number for every cortical voxel
    empty_schaefer: np.ndarray    # empty template the size of the in-brain voxels
    empty_mni: np.ndarray         # empty template the size of the flattened MNI volume


def load_brain_atlases(
        mask_path="misc/MNI152_T1_2mm_brain_mask.nii.gz",
        atlas_path="misc/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"):
    # read in 2d mni mask
    mask = nib.load(mask_path)
    og_shape = mask.shape
    mask = mask.get_fdata().flatten()
    brain_idx = np.where(mask>0)[0]

    atlas = nib.load(atlas_path)
    atlas_vec = atlas.get_fdata().flatten()
    atlas_only_brain = atlas_vec[brain_idx] # contains the schaefer parcel numbers
    cortex_vx = np.where(atlas_only_brain != 0)[0]
    schaefer_voxels = atlas_only_brain[cortex_vx]

    return BrainAtlases(
        og_shape=og_shape,
        brain_idx=brain_idx,
        atlas=atlas,
        atlas_only_brain=atlas_only_brain,
        cortex_vx=cortex_vx,
        schaefer_voxels=schaefer_voxels,
        # Making empty templates to save output
        empty_schaefer=np.zeros(atlas_only_brain.shape),
        empty_mni=np.zeros(atlas_vec.shape),
    )
