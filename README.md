# Replication package for the fMRI Recursion project
This repo contains analysis code and sample data to illustrate how we train Voxelwise Encoding Models (VEMs) based on participants' keystrokes.
*NOTE: Here we refer to the four conditions differently from the paper: RecCode -> Recursion, IterCode -> iteration, RecProse -> Prose, IterProse -> List*

## Folders and File Organization
The analysis scripts are numbered, and designed to be run in order (i.e., `1_align_keystrokes_to_vols.py`, `2_reconstruct_participant_responses.py`). We investigate the first two research questions that use neuroimaging data in the `rq1` and `rq2` scripts.

- `answer_ratings`: The authors' ratings of participants' answers on the task questions. This directory contains the three raters' scores, the script they used to rate participants' answers (`rate_answers.py`), and the script they used to review conflicting ratings together (`resolving_conflicts.py`). The consolidated ratings are in the `Consolidated_Reviews.pkl` file.

- `clean_data`: Empty for now because the preprocessed fMRI data is too large.

- `data`: Location of participants' raw keystroke and task data.

- `design_matrices`: Boolean matrices that indicate volume numbers during the scans corresponding to the four study conditions, where the value is 1 for a timepoint when a participant was working on a recursion question, and 0 otherwise, respective to each condition.

- `midprocess`: Contains the midprocessed data such as the keystrokes aligned by volumes, and the formatted keystrokes that serve as prompts to an LLM. 

- `misc`: Helper files such as brain atlases, and corresponding keys for special characters.

- `output`: Where we save the heavily processed data, such as the LLM embeddings and results from the VEMs

- `stimuli`: The original study questions and the specific questions for each block.

### Helpers
-  **`_load_brain_atlases.py`:** 
-  **`_null_distributions.py`:** Generates the null distributions for each comparison.
-  **`_results_computation.py`**
-  **`_run_participant_code.py`:** Runs participant code to measure correctness.
-  **`_test_cases.py`:** Script that contains the test cases for participant code.


## Analysis Pipeline
0. **Identify Timepoints for each condition:** (`_make_design_matrices.py`)

1. **Aligning Keystrokes to fMRI Volumes:** We first need to align the keystrokes participants typed with the corresponding volumes in the fMRI scans (`1_align_keystrokes_to_vols.py`). This works by finding an anchor point to align the two streams of data, where we use the end timestamp for the first question in a block. We then collect all the keystrokes that were typed within 1 second window of each volume acquisition. This script also formats special keys like shift (`<K:S>`), or the up arrow (`<K:U>`). If someone held down the arrow key or pressed backspace multiple times, we combine these together (`<K:U x=5>`). The final output for this step is a csv file for every scan that includes the volume numbers, the question number, and a list of keystrokes pressed during each volume (e.g., `data/102/scan_301-block_4_keystroke_df.csv`).

2. **Reconstructing Participants' Answers:** We then use these keystrokes at each volume/timepoint to reconstruct snapshots of participants' answers (`2_reconstruct_participant_responses.py`). This process was complicated in the MRI environment where some keystrokes may have been dropped due to magnetic interference, and behavior of the CodeMirror textboxes that included things like auto-indentation. We therefore use a lookahead strategy that compares the current keystrokes to the final answers participants submitted to determine whether the current keystrokes should be typed in a different location or altered slightly. The final output from this script is a Python dictionary for every fMRI scan with one entry for each volume that includes the question number, and the formatted text (e.g., `midprocess/102/scan_301-block_4-keystroke_log.pkl`).

3. **Identifying Outlier Volumes:** The MRI-safe keyboard could sometimes malfunction, where the same key would be typed repeatedly without any participant input. We identified and logged these volumes to remove (`3_identify_outlier_volumes.py`). The output for this step is a dictionary with entries specifying the volume numbers for a specific scan and participant to be removed (`misc/outlier_volumes.pkl`).

4. **Generate LLM Embeddings from Keystrokes:** This script takes in the keystroke logs from step 2, and prompts DeepSeek 6B with the snapshots of participants' answers at each timepoint (`4_create_sequential_embeddings.py`). We extract the final token from the 8th layer. The output is a Python dictionary for each scan where the keys are the formatted keystrokes, and the values are the embedding vectors, which we z-score (e.g., `embeddings/102/scan_301-block_4-keystroke_embeddings.pkl).

5. **Format Embeddings for Training VEMs:** After generating the embeddings, we need to perform a few steps to put them into a suitable format for training VEMs. We first remove the timepoints associated with outlier volumes identified in Step 3, then we identify and concatenate embeddings associated with each of our four study conditions: RecCode, IterCode, RecProse, IterProse (`5_prepare_embeddings_for_fir.py`). We apply PCA to reduce computational costs of training VEMs such that 99% of the variance is retained, then we add an additional regressor to model the number of keystrokes that a participant typed at each timepoint. Finally, we create 10 copies of this matrix, delayed by 1 to 10 timepoints. The output for this step is a stimulus matrix saved as a pickle file for each condition, per participant (`output/stimulus_matrices/102-iteration-stimulus_matrix.pkl`).

*Note: Without the fMRI data, Steps 6 and 7 below will not run. We include the final, organized output file after running Step 7 here (`organized_results_by_condition.pkl`).*

6. **Training VEMs:** Training a VEM that models brain activity (measured as BOLD signal) from a stimulus matrix of LLM embeddings (from the previous step) is based on the signal processing technique Finite Impulse Response (FIR) used in neuroimaging analysis (`6_fir.py`). Essentially, we learn a linear combination of values in the LLM embeddings that equals values of BOLD signal recorded in each voxel. We first format the fMRI data in the same way that we formatted the LLM embeddings regarding the outlier volumes and timepoints specific to each of the four study conditions. For a given participant and condition, we then split both the fMRI data ($Y$) and the stimulus matrix ($X$) into a training set (first 90%) and test set (last 10%), then optimize the VEM ($\beta$), which is a matrix of weights, using ridge regression. On the training data we perform 5 cross validation splits and test 12 different values for a hyperparameter $\alpha$ that determines the strength of the penalty for large weights. After finding the best-performing $\alpha$ value, we then use the trained encoding model $\beta$ to predict the test set fMRI data from the test set stimulus matrix. The output from running this step is (1) a statistical map of the brain where the value in each voxel is the correlation between the predicted and recorded signal (`output/VEMs/102/iteration/correlations.pkl`) and (2) the model weights (`output/VEMs/102/iteration/model_weights.pkl`).

7. **Organizing Results:** After training the VEMs and testing their performance on a held-out test set, we then identify the voxels where modeling performed best, and collect these across conditions and participants (`7_organize_top_performing_voxels_and_parcels.py`). To identify voxels that are modeled best, we identify those where the correlation between the predicted and recorded signal is above chance levels. For this we use permutation testing, where in this particular case, we circularly shift the predicted signal by a random amount 1000 times and recompute its correlation to the test signal. This process builds a null distribution against which we compare our observed correlation value and calculate a p-value. We then correct for multiple comparisons using Benjamini-Hochberg ($q<0.05$). We calculate the average number of significant voxels per participant and per condition, and use this number ($6{,}525$) as our threshold for examining the top-modeled participants across participants and conditions. We also identify the locations in the brain where these voxels are located using the Schaefer Atlas as a reference. The output for this step is a Python dictionary saved as a Pickle file (`output/organized_results_by_condition.pkl`).

## Research Questions
**RQ1: How does the neural basis of recursive programming compare to that of iterative programming?** (`rq1_reccode_vs_itercode.ipynb`)
- We investigate differences between recursive and iterative programming by examining set intersections and differences in each participant between their top-modeled voxels for the two conditions. We perform these steps, as well as permutation testing and correction for multiple comparisons in the script. Results are plotted onto brains using pycortex, which opens as a locally hosted webpage

**RQ2: How does recursion in programming relate to recursion in natural language?** (`rq2_reccode_vs_recprose.ipynb`)
- We perform nearly the same steps as those in the first research question for comparing recursion in code to recursion in natural language. For this part though, to isolate the recursive aspects of code and recursive aspects of prose, we first take the set differences between these conditions and their iterative counterparts. That is, we compute RecCode' as the set difference of (RecCode - IterCode), and RecProse' as (RecProse - IterProse). Results are plotted onto brains using pycortex, which opens as a locally hosted webpage.

**RQ3: What are programmers' strategies for recursive programming?** (Codebook_Final.xlsx)


