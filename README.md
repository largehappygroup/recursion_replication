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

5. **Format Embeddings for Training VEMs:** After generating the embeddings, we need to perform a few steps to format them as a 2D matrix that we use for training the VEMs





