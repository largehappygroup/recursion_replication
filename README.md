# Replication package for the fMRI Recursion project
This repo contains analysis code and sample data to illustrate how we train Voxelwise Encoding Models (VEMs) based on participants' keystrokes.
Also included are scripts for calculating study-specific statistics, like pilot study completion times.

## Raw Data
Our raw data consists of the recorded fMRI data (too large to include here), participants' keystrokes (recorded as ASCII codes with associated timestamps), and participants' final responses on each of the questions.

## Analysis Pipeline
1. **Aligning Keystrokes to fMRI Volumes:** We first need to align the keystrokes participants typed with the corresponding volumes in the fMRI scans (1_align_keystrokes_to_vols.py). This works by finding an anchor point to align the two streams of data, where we use the end timestamp for the first question in a block. We then collect all the keystrokes that were typed within 1 second window of each volume acquisition. This script also formats special keys like shift (<K:S>), or the up arrow (<K:U>). If someone held down the arrow key or pressed backspace multiple times, we combine these together (<K:U x=5>). The final output for this step is a csv file for every scan that includes the volume numbers, the question number, and a list of keystrokes pressed during each volume (e.g., 102/scan_301-block_4_keystroke_df.csv).

2. **Reconstructing Participants' Answers:** We then use these keystrokes at each volume/timepoint to reconstruct snapshots of participants' answers (2_reconstruct_participant_responses.py). This process was complicated in the MRI environment where some keystrokes may have been dropped due to magnetic interference, and behavior of the CodeMirror textboxes that included things like auto-indentation. We therefore use a lookahead strategy that compares the current keystrokes to the final answers participants submitted to determine whether the current keystrokes should be typed in a different location or altered slightly. The final output from this script is a Python dictionary for every fMRI scan with one entry for each volume that includes the question number, and the formatted text (e.g., 102/scan_301-block_4-keystroke_log.pkl).

3. **Identifying Outlier Volumes:** The MRI-safe keyboard could sometimes malfunction, where the same key would be typed repeatedly without any participant input. We identified and logged these volumes to remove (3_identify_outlier_volumes.py). The output for this step is a dictionary with entries specifying the volume numbers for a specific scan and participant to be removed (misc/outlier_volumes.pkl).

4. **Generate LLM Embeddings from Keystrokes:** This script takes in the keystroke logs from step 2, and prompts DeepSeek 6B with the snapshots of participants' answers at each timepoint (4_create_sequential_embeddings.py). We extract the final token from the 8th layer. The output is a Python dictionary for each scan where the keys are the formatted keystrokes, and the values are the embedding vectors (which we z-score).

5. **Format Embeddings for Training VEMs:** After generating the embeddings, we need to perform a few steps to put them into a suitable format for training VEMs. We first remove the timepoints associated with outlier volumes identified in Step 3, then we identify and concatenate embeddings associated with each of our four study conditions (RecCode, IterCode, RecProse, IterProse). We apply PCA to reduce computational costs of training VEMs such that 99% of the variance is retained, then we add an additional regressor to model the number of keystrokes that a participant typed at each timepoint. Finally, we create 10 copies of this matrix, delayed by 1 to 10 timepoints. The output for this step is a stimulus matrix saved as a pickle file for each condition, per participant (stimulus_matrices/102-itercode-stimulus_matrix.pkl).

6. **Training VEMs:** Training a VEM that models brain activity (measured as BOLD signal) from a stimulus matrix of LLM embeddings (from the previous step) is based on the signal processing technique Finite Impulse Response (FIR) used in neuroimaging analysis (6_fir.py). Essentially, we learn a linear combination of values in the LLM embeddings that equals values of BOLD signal recorded in each voxel. We first format the fMRI data in the same way that we formatted the LLM embeddings regarding the outlier volumes and timepoints specific to each of the four study conditions. For a given participant and condition, we then split both the fMRI data ($Y$) and the stimulus matrix ($X$) into a training set (first 90%) and test set (last 10%), then optimize the VEM ($\beta$), which is a matrix of weights, using ridge regression. On the training data we perform 5 cross validation splits and test 12 different values for a hyperparameter $\alpha$ that determines the strength of the penalty for large weights. After finding the best-performing $\alpha$ value, we then use the trained encoding model $\beta$ to predict the test set fMRI data from the test set stimulus matrix. The output from running this step is (1) a statistical map of the brain where the value in each voxel is the correlation between the predicted and recorded signal (VEMs/102/itercode/correlations.pkl) and (2) the model weights (VEMs/102/itercode/model_weights.pkl).

7. **Organizing Results:** After training the VEMs and testing their performance on a held-out test set, we then identify the voxels where modeling performed best using permutation testing. That is, we circularly shift the predicted signal by a random amount 1000 times and recompute its correlation to the test signal. This process builds a null distribution against which we compare our observed correlation value and calculate a p-value. We then correct for multiple comparisons. We also identify the locations of these voxels in the Schaefer Atlas

## Research Questions

## Helpers





