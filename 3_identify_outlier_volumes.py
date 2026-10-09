import os
import re
import ast
import glob
import pickle
import numpy as np
import pandas as pd


participants = os.listdir("midprocess")

def find_root(key):
    if re.search(r'<*(\d+)>', key):
        return re.sub(r'\sx=(\d+)>', '', key)
    elif re.search(r'^<K:+(?!\s)', key):
        return re.sub(r'>', '', key)
    else:
        return key

def identify_outliers(p, keys, characters):
    mean = np.mean(keys)
    extreme_val = np.std(keys) * 2
    outlier_loc = list(np.where(keys > extreme_val + mean)[0])
    outlier_loc = [int(i) for i in outlier_loc]

    filtered_outliers = []
    for idx in outlier_loc:
        keys_pressed = ast.literal_eval(characters.loc[idx, 'keystrokes'])
        if len(keys_pressed) <= 2: # finding keys pressed that aren't just repeated backspaces or arrow keys
            continue
        elif extreme_val - len(keys_pressed) > 0: # another check for keystrokes that are mostly non-printable keys
            continue
        else: # fishy keystrokes, but care about keystrokes that are from hardware issues where keys just get repeated
            unique_keys = {find_root(k) for k in keys_pressed}
            if len(unique_keys) <= 2: # finding sequences that are repeated keystrokes
                filtered_outliers.append(idx)
    return filtered_outliers

outlier_vols = {p : {} for p in participants}
for p in participants:
    # p = '108'
    for scan_block in glob.glob(f"midprocess/{p}/*-num_keystrokes_regressor.pkl"):
        try:
            # with open(f"midprocess/{p}/num_keystrokes_regressor.pkl", 'rb') as f:
            with open(scan_block, 'rb') as f:
                keys = pickle.load(f)
        except:
            continue

        scan_info = (scan_block.split('/')[-1]).removesuffix('-num_keystrokes_regressor.pkl')

        # cross reference with keystrokes file
        characters_file = f"midprocess/{p}/{scan_info}_keystroke_df.csv"
        characters = pd.read_csv(characters_file)

        filtered_outliers = identify_outliers(p, keys, characters)
        if filtered_outliers:
            outlier_vols[p][scan_info] = [int(i) for i in filtered_outliers]
        # break
    # break


outpath = "misc/outlier_volumes.pkl"

print(outlier_vols)

with open(outpath, 'wb') as f:
    pickle.dump(outlier_vols, f)
