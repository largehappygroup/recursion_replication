import os
import pickle
import subprocess
import numpy as np
import pandas as pd

# from reconstruct_participant_responses import get_question_text

DATA_DIR = "./data"

# task-csv 'task' abbreviation -> condition name
TASK_CONDITIONS = {
    'rec': 'recursion',
    'it': 'iteration',
    'pro': 'prose',
    'lis': 'list',
}

conditions = {
    'recursion': pd.read_csv("stimuli/recursion.csv"),
    'iteration': pd.read_csv("stimuli/iteration.csv"),
    'prose'    : pd.read_csv("stimuli/prose.csv"),
    'list'     : pd.read_csv("stimuli/list.csv")
}
stimuli = pd.concat(conditions.values(), ignore_index=True)

# results from running participants' code against test cases
code_correctness = pd.read_csv("code_test_results.csv")

# only prose/list responses are rated alongside their question text -
# recursion/iteration responses are code completions, so the question
# (the scaffolding) isn't needed to judge the response
NEEDS_QUESTION_TEXT = {'prose', 'list'}

try: # if the rater quit and is restarting, read in their progress
    with open("task_rating_progress.pkl", "rb") as f:
        progress = pickle.load(f)
    print("Welcome back!")
    print(progress)
except: # if they're starting fresh, progress will be empty
    progress = {}
    print("Starting Fresh")

output = progress.copy() # updating output with progress, whether it's empty or not

def get_question_text(question_num):
    question_idx = np.where(stimuli['stim_id'] == question_num)
    question_text = (list(stimuli.loc[question_idx, 'content']))[0]
    return question_text

# given the participant's row, print their response (and question, if needed) to screen
def show_response(pid, count, row, condition):
    print(f"Participant {pid} Response {count} ({condition}):\n")
    if condition in NEEDS_QUESTION_TEXT:
        question_text = get_question_text(int(row['stim_id']))
        print(f"Question: {question_text}\n")
    print(row['participant_output'],'\n')
    print("(1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree), 'q' to quit")

# getting ratings from rater in command line
def get_answers(condition, pid, stim_id): # always giving the option to quit
    
    if condition in['prose', 'list']:    
        correct = input("The answer is correct: ")
        efficient = ''
        if correct == 'q':
            return 'q'
    
    elif condition in ['recursion', 'iteration']:
        stim_row = np.where((code_correctness['pid'] == int(pid)) & (code_correctness['stim_id'] == int(stim_id)))[0][0]
        tests_passed = int(code_correctness.loc[stim_row, 'tests_passed'])
        total_tests  = int(code_correctness.loc[stim_row, 'tests_total'])
        correct = tests_passed/total_tests
        print(f"Test cases passed: {tests_passed}/{total_tests} = {correct}")
        efficient = input("The answer uses an efficient approach: ")
        if efficient == 'q':
            return 'q'
    
    complete = input("The answer completely satisfies the question: ")
    if complete == 'q':
        return 'q'

    readable = input("The response is clearly written and formatted (i.e., following question guidelines): ")
    if readable == 'q':
        return 'q'

    return [correct, efficient, complete, readable]

def main():
    pids = sorted(f.removesuffix('_task.csv') for f in os.listdir(DATA_DIR))
    for pid in pids: # for each participant
        
        df = pd.read_csv(f"{DATA_DIR}/{pid}_task.csv")
        df = df[df['stim_id'] != 'stim_id'].reset_index(drop=True) # drop repeated header rows from concatenated blocks
        curr_participant = {}
        count = 0
        for i, row in df.iterrows(): # for each question this participant answered
            condition = TASK_CONDITIONS.get(row['task'])
            if condition is None:
                continue
            key = f"{row['stim_name']}_{int(row['stim_id'])}"
            if pid in output and key in output[pid]: # if rater has already done this, update and add to dictionary
                print(f"already finished {key} for {pid}")
                curr_participant[key] = output[pid][key]
                count += 1
                continue

            show_response(pid, count, row, condition)
            answers = get_answers(condition, pid, int(row['stim_id']))
            if answers == 'q':
                output[pid] = curr_participant.copy()
                with open("task_rating_progress.pkl", "wb") as f:
                    pickle.dump(output, f)
                return

            elif len(answers) == 4:
                correct, efficient, complete, readable = answers[0], answers[1], answers[2], answers[3]
            else:
                print("something wrong happened")
            curr_participant[key] = [correct, efficient, complete, readable]
            count += 1
            subprocess.run(['clear'])
        output[pid] = curr_participant

    with open("Task_Response_Ratings.pkl", "wb") as f:
        pickle.dump(output, f)

    subprocess.run(['clear'])
    print("All finished! Thank you for your service.")

if __name__ == "__main__":
    main()
