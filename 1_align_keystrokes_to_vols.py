import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

##########################################################################################
############# VARIABLES ##################################################################
##########################################################################################

character_path = f"misc/special_character_symbols.pkl"
with open(character_path, 'rb') as f:
    special_characters = pickle.load(f)

# maps, per participant, each task-info section (scan_order, chronological) to the
# actual scan number it was recorded under -- built by organize_and_align_files.py.
# needed because a block that was restarted (e.g. after a technical issue) produces
# more than one section for the same block_num, so block_num alone can't tell them apart.
blocks_to_scans_path = "misc/blocks_to_scans_mapping.pkl"
with open(blocks_to_scans_path, 'rb') as f:
    blocks_to_scans = pickle.load(f)

##########################################################################################
############# PARTICIPANT CLASS ##########################################################
##########################################################################################

class Participant:

    def __init__(self, pid):
        self.pid = pid
        self.data_dir = f"data/{pid}"

        self.task_info = self._load_task_info()
        self.block_order = self._load_block_order()
        self.block_info = split_task_info_into_blocks(self.task_info) if self.task_info is not None else {}
        self.scan_to_attempt = self._load_scan_to_attempt_map()
        self.designs = self._load_designs()
        self.keystrokes = self._load_keystrokes()

    def _load_task_info(self):
        try:
            return pd.read_csv(f"{self.data_dir}/{self.pid}_task.csv")

        except FileNotFoundError:
            print(f"No task info for Participant {self.pid}.")
            return None

    def _load_block_order(self):
        try:
            df = pd.read_csv(f"{self.data_dir}/block_order.csv")
            return list(df['block'])
        except FileNotFoundError:
            print(f"No block order for Participant {self.pid}.")
            return None

    def _load_scan_to_attempt_map(self):
        mapping = blocks_to_scans.get(int(self.pid), blocks_to_scans.get(self.pid))
        if not mapping:
            print(f"No scan-to-attempt mapping for Participant {self.pid}.")
            return {}
        return {info['scan_num']: scan_order for scan_order, info in mapping.items()}

    def _load_designs(self):
        design_dir = f"design_matrices/{self.pid}"
        try:
            files = os.listdir(design_dir)
        except FileNotFoundError:
            print(f"No design matrices for Participant {self.pid}.")
            return {}
        return {f.removesuffix('.csv'): pd.read_csv(f"{design_dir}/{f}") for f in files}

    def _load_keystrokes(self):
        keystroke_dir = f"{self.data_dir}/keystrokes"
        try:
            files = os.listdir(keystroke_dir)
        except FileNotFoundError:
            print(f"No keystrokes for Participant {self.pid}.")
            return {}
        
        if not self.block_order:
            return None

        nested = defaultdict(dict)
        for f in files:
            m = re.match(r'block_(\d+)_question_(\d+)', f)
            block_num, question_num = int(m.group(1)), int(m.group(2))
            nested[block_num][question_num] = pd.read_csv(f"{keystroke_dir}/{f}", header=None, names = ['pid', 'question_label', 'question_num', 'ascii_code', 'timestamp'])
        return dict(nested)
    
##########################################################################################
############# FUNCTIONS ##################################################################
##########################################################################################

def split_task_info_into_blocks(task_info):
    is_header_row = task_info["participant_id"] == "participant_id"
    task_info = task_info.copy()
    task_info["scan_order"] = is_header_row.cumsum()

    # group by scan_order (one group per task-info section, in chronological order),
    # not block_num -- a restarted block produces two sections with the same block_num,
    # and those must stay separate or their end_time/stim rows get mixed together.
    blocks = {
		scan_order: g.drop(columns="scan_order").reset_index(drop=True)
		for scan_order, g in task_info[~is_header_row].groupby("scan_order")
	}
    return blocks

def find_early_keystrokes(df):
    first_idx = df['stim_id'].first_valid_index()
    early_keystrokes = (df.loc[:first_idx-1, "keystrokes"]).sum()
    return first_idx, early_keystrokes

def concat_duplicates(key_list):
    nonprintable_keys = [    
        '<K:S>', # shift
        '<K:BS>', # backspace
        '<K:CTRL>', # ctrl
        '<K:CTRLT>', # new tab shortcut
        '<K:ESC>', # escape
        '<K:L>', # left arrow
        '<K:R>', # right arrow
        '<K:U>', # up arrow
        '<K:D>', # down arrow
    ]
    
    concatenated_keys = []
    # iterate through the list of keystrokes
    prev_key = ''
    prev_key_count = 0
    for i,key in enumerate(key_list):
        
        # if key doesn't equal prev key - 
        # i.e. an incremented non-printable key that should be recorded or a new key
        if key != prev_key:
            
            # if there's more than one occurrence
            # format in special way
            if prev_key_count > 1:
                new_entry = f"{prev_key[:-1]} x={prev_key_count}>"
                concatenated_keys.append(new_entry)
            
            # otherwise just append key
            elif prev_key_count == 1:
                concatenated_keys.append(prev_key)
            
            # reset variables
            prev_key = ''
            prev_key_count = 0
        
        # if it's just a regular key, append it to the output
        # then reset variables
        if key not in nonprintable_keys:
            concatenated_keys.append(key.lower())
            prev_key = ''
            prev_key_count = 0
            continue
        
        # if it's a non-printable key
        else:
            # increment if it's the same as previous key
            if key == prev_key:
                prev_key_count += 1
            else: # if it's a new key, start sequence here
                prev_key = key
                prev_key_count = 1
        
        # if we reach the end of the list and have a nonempty entry, record it
        if i == len(key_list) - 1 and prev_key_count > 0:
            if prev_key_count == 1:
                concatenated_keys.append(prev_key)
            elif prev_key_count > 1:
                new_entry = f"{prev_key[:-1]} x={prev_key_count}>"
                concatenated_keys.append(new_entry)
    
    return concatenated_keys

def process_keystrokes(ascii_keystrokes):
    
    # converting ascii into characters
    keystroke_chars = [chr(asci) for asci in ascii_keystrokes]

    # converting special ascii characters for things like enter and shift 
    converted_chars = [special_characters[char] if char in special_characters.keys() else char for char in keystroke_chars]

    concatenated_chars = concat_duplicates(converted_chars)
    return concatenated_chars


def find_volume_keystrokes(design_matrix, keystrokes, block):
    keystrokes_by_volume = []
    
    keystrokes['timestamp'] = pd.to_datetime(keystrokes['timestamp'])
    start_time = pd.Timestamp(design_matrix.loc[0,'timestamp']) - pd.Timedelta(seconds=1)
    for v in range(0,len(design_matrix)):

        end_time = pd.Timestamp(design_matrix.loc[v, 'timestamp'])
        stim_name, stim_id = design_matrix.loc[v, 'stim_name'], design_matrix.loc[v, 'stim_id']
        
        idx_keystrokes_in_window = np.where(keystrokes['timestamp'].between(start_time, end_time))[0]
        ascii_keystrokes = list(keystrokes.loc[idx_keystrokes_in_window, 'ascii_code'])
        cleaned_keystrokes = process_keystrokes(ascii_keystrokes)
        
        new_entry = [v, block, stim_id, stim_name, cleaned_keystrokes]
        keystrokes_by_volume.append(new_entry)
        # print(f"Vol {v} | Stim Name {stim_name} | Start {start_time} | End {end_time} | Keys {cleaned_keystrokes}")

        start_time = end_time
    
    keystroke_df = pd.DataFrame(keystrokes_by_volume, columns=["vol_num", "block", "stim_id", "stim_name", "keystrokes"])
    first_idx,early_keystrokes = find_early_keystrokes(keystroke_df)
    keystroke_df.at[first_idx, 'keystrokes'] = list(early_keystrokes) + list(keystroke_df.at[first_idx, 'keystrokes'])
    return keystroke_df
   
def add_question_context(design, this_block):
    design = design.copy()
    task_cols = ['recursion', 'iteration', 'prose', 'list']
    active = design[task_cols].sum(axis=1) > 0
    
    group_id = (active & ~active.shift(fill_value=False)).cumsum()
    group_id = group_id.where(active)
    
    block = this_block.sort_values('end_time').reset_index(drop=True)
    block.index += 1
    
    design = design.merge(block[['stim_id', 'stim_name']], left_on=group_id, right_index=True, how='left')
    return design

def find_question_anchor_points(design, this_block):
    diffs = design.diff(axis=0).fillna(0)
    transitions = diffs['recursion'] + diffs['iteration'] + diffs['prose'] + diffs['list']

    # the -1 transitions for when a question ends need to be moved back by one index
    question_ends = list(np.where(transitions == -1)[0]-1)
    
    # for scans that ended mid-question
    if len(question_ends) == 0 or design.loc[len(design)-1, ['recursion', 'iteration', 'prose', 'list']].sum() == 1: # some participants who  
        question_ends.append(len(design)-1)
    question_ends = [int(el) for el in question_ends]

    ref_index = question_ends[0]
    ref_timestamp = datetime.strptime(this_block.loc[0,'end_time'], "%Y-%m-%d %H:%M:%S.%f")

    return ref_index, ref_timestamp

def format_question_keystrokes(pid, block):
    keys = pid.keystrokes[int(block)]
    sorted_questions = sorted(keys.keys())
    keys = {k: keys[k] for k in sorted_questions}
    all_keys = pd.concat(keys.values(), axis=0, ignore_index=True)
    return all_keys

def process_participant(p):
    bass_outpath = f"midprocess/{p.pid}"
    if not os.path.isdir(bass_outpath):
        os.mkdir(bass_outpath)
    # the output should be for each scan/block, the keystrokes aligned with the vol numbers
    for scan_block, design in p.designs.items():
        print(scan_block)
        scan_str, block = scan_block.split('-')
        scan_num = int(scan_str.removeprefix('scan_'))
        block = block.split('_')[-1]

        scan_order = p.scan_to_attempt.get(scan_num)
        if scan_order is None:
            print(f"No task-info section found for scan {scan_num} (Participant {p.pid}), skipping.")
            continue
        this_block = p.block_info[scan_order]

        all_keys = format_question_keystrokes(p, block)
        ref_index,ref_timestamp = find_question_anchor_points(design, this_block)
        design['timestamp'] = ref_timestamp + (design.index - ref_index) * pd.Timedelta(seconds=1)
        design = add_question_context(design, this_block)
        keystroke_df = find_volume_keystrokes(design, all_keys, block)
        
        outpath = f"{bass_outpath}/{scan_block}_keystroke_df.csv"
        keystroke_df.to_csv(outpath, index=False)
        # break
        
        

# The purpose of the main function is to iterate through each participants' keystroke files
# and figure out what keys were pressed during what volumes of the fMRI scan
# The output should be in a format that can be ingested by a method for creating model embeddings
# maybe a dictionary where keys are volume numbers, and keystrokes are the accumulated answer at that points
def main():
    participants = os.listdir(f"data")
    for pid in participants:
        print(pid)
        p = Participant(pid)
        if p.designs == {} or not p.block_order:
            continue

        process_participant(p)
        # break
    
if __name__ == "__main__":
    main()
