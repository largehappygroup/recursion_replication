import pickle
import subprocess
import numpy as np
import pandas as pd

# THIS SCRIPT WALKS A HUMAN RESOLVER THROUGH ONLY THE (participant, question, criterion)
# COMBINATIONS THAT inter_rater_agreement.py FLAGGED AS needs_review IN Consolidated_Reviews.pkl -
# EITHER BECAUSE THE THREE RATERS' VALENCES (NEGATIVE/NEUTRAL/POSITIVE) DISAGREED, OR BECAUSE OF A
# DATA ISSUE (MISSING/NON-NUMERIC/OUT-OF-RANGE VALUE). IT SHOWS THE ORIGINAL RESPONSE (AND QUESTION
# TEXT, WHERE RELEVANT) JUST LIKE rate_answers.py, BUT SKIPS STRAIGHT PAST EVERYTHING THAT'S ALREADY
# RESOLVED AND ONLY ASKS ABOUT THE FLAGGED CRITERIA FOR EACH FLAGGED RESPONSE.
#
# Consolidated_Reviews.pkl is the only file this script reads and writes - resolving an item sets
# its needs_review flag to False right there, so re-running the script (e.g. after quitting with
# 'q') naturally skips everything already resolved, with no separate progress file needed.

DATA_DIR = "./data"
CONSOLIDATED_PATH = "Consolidated_Reviews.pkl"

# task-csv 'task' abbreviation -> condition name
TASK_CONDITIONS = {
    'rec': 'recursion',
    'it': 'iteration',
    'pro': 'prose',
    'lis': 'list',
}

# only prose/list responses are rated alongside their question text - recursion/iteration
# responses are code completions, so the question (the scaffolding) isn't needed to judge the
# response
NEEDS_QUESTION_TEXT = {'prose', 'list'}

FIELD_PROMPTS = {
    'correct': "The answer is correct: ",
    'efficient': "The answer uses an efficient approach: ",
    'complete': "The answer completely satisfies the question: ",
    'readable': "The response is clearly written and formatted (i.e., following question guidelines): ",
}

conditions = {
    'recursion': pd.read_csv("stimuli/recursion.csv"),
    'iteration': pd.read_csv("stimuli/iteration.csv"),
    'prose'    : pd.read_csv("stimuli/prose.csv"),
    'list'     : pd.read_csv("stimuli/list.csv")
}
stimuli = pd.concat(conditions.values(), ignore_index=True)


def get_question_text(question_num):
    question_idx = np.where(stimuli['stim_id'] == question_num)
    question_text = (list(stimuli.loc[question_idx, 'content']))[0]
    return question_text


# builds pid -> key -> [fields] for every criterion that's still flagged needs_review
def build_pending(consolidated):
    pending = {}
    for pid, keys in consolidated.items():
        for key, fields in keys.items():
            still_pending = [field for field, entry in fields.items() if entry['needs_review']]
            if still_pending:
                pending.setdefault(pid, {})[key] = still_pending
    return pending


def show_response(pid, key, row, condition):
    print(f"Participant {pid} - {key} ({condition}):\n")
    if condition in NEEDS_QUESTION_TEXT:
        question_text = get_question_text(int(row['stim_id']))
        print(f"Question: {question_text}\n")
    print(row['participant_output'], '\n')

# prompts for a resolved value for one flagged field, re-prompting on invalid (non-numeric,
# non-'q') input, and returns 'q' if the resolver wants to quit
def get_resolution(field, condition):
    if field == 'correct' and condition in ('recursion', 'iteration'):
        prompt = "Raters disagreed on the computed test result - enter the correct tests_passed/tests_total fraction: "
    else:
        prompt = FIELD_PROMPTS[field]

    while True:
        answer = input(prompt)
        if answer == 'q':
            return 'q'
        try:
            return float(answer)
        except ValueError:
            print(f"'{answer}' isn't a number, try again ('q' to quit)")


def resolve(entry, value):
    entry['original_reason'] = entry['reason']
    entry['reason'] = 'manually_resolved'
    entry['needs_review'] = False
    entry['value'] = value


def save_consolidated(consolidated):
    with open(CONSOLIDATED_PATH, "wb") as f:
        pickle.dump(consolidated, f)


def main():
    with open(CONSOLIDATED_PATH, "rb") as f:
        consolidated = pickle.load(f)

    pending = build_pending(consolidated)
    n_remaining = sum(len(fields) for keys in pending.values() for fields in keys.values())
    print(f"{n_remaining} criteria left to review\n")

    for pid in sorted(pending.keys(), key=int):
        df = pd.read_csv(f"{DATA_DIR}/{pid}_task.csv")
        df = df[df['stim_id'] != 'stim_id'].reset_index(drop=True)  # drop repeated header rows from concatenated blocks

        for _, row in df.iterrows():
            key = f"{row['stim_name']}_{int(row['stim_id'])}"
            if key not in pending[pid]:
                continue

            condition = TASK_CONDITIONS.get(row['task'])
            show_response(pid, key, row, condition)
            print("(1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree), 'q' to quit")

            for field in pending[pid][key]:
                entry = consolidated[pid][key][field]
                answer = get_resolution(field, condition)
                if answer == 'q':
                    save_consolidated(consolidated)
                    return
                resolve(entry, answer)

            save_consolidated(consolidated)  # persist after each response, in case of a crash mid-session
            subprocess.run(['clear'])

    print("All finished! No conflict remains.")


if __name__ == "__main__":
    main()
