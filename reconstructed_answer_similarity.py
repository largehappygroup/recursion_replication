import os
import sys
import glob
import difflib
import contextlib
import pandas as pd
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import reconstruct_participant_responses as recon
recon = import_module('2_reconstruct_participant_responses')


def final_text_per_question(log):
    """Last full_text entry per stim_id, in volume order, from a process_keystrokes log."""
    last = {}
    for i in sorted(log.keys()):
        e = log[i]
        sid = e['stim_id']
        if sid is not None:
            last[sid] = e['full_text']
    return last


def similarity(reconstructed_text, oracle_text):
    """difflib.SequenceMatcher ratio (0-1) between reconstructed and oracle text."""
    return difflib.SequenceMatcher(None, reconstructed_text, oracle_text).ratio()


def fragmentation_score(reconstructed_text, oracle_text, small_edit_max=5):
    """Counts scattered small (<= small_edit_max char) edits between reconstructed and
    oracle text, via difflib's opcodes. One big replace/insert block is a normal
    correction; many small, scattered edits despite a high overall similarity() ratio
    is the signature of stray/misplaced characters sitting inside otherwise-correct
    text, which whole-text similarity alone hides."""
    matcher = difflib.SequenceMatcher(None, reconstructed_text, oracle_text, autojunk=False)
    small_edits = 0
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        if max(a2 - a1, b2 - b1) <= small_edit_max:
            small_edits += 1
    return small_edits


def score_answers(log, oracle_answers):
    """Per-stim_id similarity ratios between a log's final answers and their oracles."""
    ft = final_text_per_question(log)
    return {
        sid: similarity(ft[sid], oracle)
        for sid, oracle in oracle_answers.items()
        if sid in ft
    }


def score_participant(person, align_with_oracle=True, verbose=False):
    """Runs the reconstruction pipeline for every keystroke file a participant has,
    scoring each question's reconstructed answer against its oracle. Returns
    {stim_id: ratio}, keeping the highest-volume-index result if a stim_id
    appears in more than one keystroke file."""
    oracle_answers, _ = recon.load_task_metadata(person)
    if not oracle_answers:
        raise ValueError(f"no task file / oracle answers found for participant {person}")

    participant_path = f"midprocess/{person}"
    scores = {}
    for f in sorted(glob.glob(f"{participant_path}/*_keystroke_df.csv")):
        # process_keystrokes unconditionally announces each correction it makes;
        # hide that unless the caller wants to see it.
        redirect = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(open(os.devnull, 'w'))
        with redirect:
            log = recon.process_keystrokes(f, person, align_with_oracle=align_with_oracle, save_log=False)
        scores.update(score_answers(log, oracle_answers))
    return scores


def fragmentation_report(person, align_with_oracle=True, verbose=False, small_edit_max=5):
    """Like score_participant, but pairs each question's similarity ratio with its
    fragmentation_score. Returns {stim_id: (ratio, fragmentation)}."""
    oracle_answers, _ = recon.load_task_metadata(person)
    if not oracle_answers:
        raise ValueError(f"no task file / oracle answers found for participant {person}")

    participant_path = f"midprocess/{person}"
    report = {}
    for f in sorted(glob.glob(f"{participant_path}/*_keystroke_df.csv")):
        redirect = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(open(os.devnull, 'w'))
        with redirect:
            log = recon.process_keystrokes(f, person, align_with_oracle=align_with_oracle, save_log=False)
        ft = final_text_per_question(log)
        for sid, oracle in oracle_answers.items():
            if sid in ft:
                report[sid] = (similarity(ft[sid], oracle), fragmentation_score(ft[sid], oracle, small_edit_max))
    return report


def _stim_label(sid):
    row = recon.stimuli[recon.stimuli['stim_id'] == sid]
    name = row['name'].iloc[0] if len(row) else '???'
    return f"{sid}_{name}"


if __name__ == "__main__":
    align_with_oracle = True
    verbose = False
    csv_path = f"reconstructed_code_fidelity.csv"

    datapath = f"midprocess"
    participants = sorted(os.listdir(datapath))

    rows = {}
    for person in participants:
        # person = '120'

        try:
            scores = score_participant(person, align_with_oracle=align_with_oracle, verbose=verbose)
        except ValueError as e:
            print(e)
            continue

        print(f"=== participant {person} ===")
        for sid in sorted(scores):
            print(f"{_stim_label(sid):30s} {scores[sid]:.4f}")
        if scores:
            mean = sum(scores.values()) / len(scores)
            print(f"{'mean':30s} {mean:.4f}  ({len(scores)} questions)")
        print()

        rows[person] = {_stim_label(sid): round(ratio, 4) for sid, ratio in scores.items()}

        # break

    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'participant'
    df = df[sorted(df.columns, key=lambda c: int(c.split('_')[0]))]
    df = df.sort_index()
    df.to_csv(csv_path)
    print(f"wrote {csv_path} ({len(df)} participants x {len(df.columns)} questions)")
