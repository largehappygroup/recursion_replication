import os
import re
import sys
import json
import tempfile
import subprocess
import pandas as pd

DATA_DIR = "../data"
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_code_test_runner.py")
TIMEOUT_SECONDS = 5

# symmetric tree (see comments on stim_id 106/207):
#        5
#      3   3
#     2 1 1 2
SYMMETRIC_TREE = [5, [3, [2, None, None], [1, None, None]], [3, [1, None, None], [2, None, None]]]
ASYMMETRIC_TREE = [5, [3, [2, None, None], [1, None, None]], [3, [1, None, None], [9, None, None]]]

# a few hand-picked test cases per recursion/iteration question, keyed by
# stim_id. Expected values are our own correct reference outputs, not the
# stimuli files' 'answer' column - that column is researcher-authored seed
# code participants were asked to reproduce from memory, and is sometimes
# deliberately buggy (see rec_symmetric/rec_intPalindrome), not an oracle.
PROBLEMS = {
    # --- recursion ---
    100: {  # rec_fibonacci
        'func': 'fibonacci',
        'cases': [
            {'kind': 'args', 'args': [0], 'expected': 0},
            {'kind': 'args', 'args': [1], 'expected': 1},
            {'kind': 'args', 'args': [5], 'expected': 5},
            {'kind': 'args', 'args': [10], 'expected': 55},
        ],
    },
    101: {  # rec_palindrome
        'func': 'isPalindrome',
        'cases': [
            {'kind': 'args', 'args': ['racecar', 0, 6], 'expected': True},
            {'kind': 'args', 'args': ['hello', 0, 4], 'expected': False},
            {'kind': 'args', 'args': ['a', 0, 0], 'expected': True},
            {'kind': 'args', 'args': ['ab', 0, 1], 'expected': False},
        ],
    },
    102: {  # rec_permutations (prints, doesn't return)
        'func': 'permute',
        'cases': [
            {'kind': 'stdout', 'args': ['ABC', 0, 2], 'expected': sorted(['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA'])},
            {'kind': 'stdout', 'args': ['AB', 0, 1], 'expected': sorted(['AB', 'BA'])},
        ],
    },
    103: {  # rec_reverse
        'func': 'reverseString',
        'cases': [
            {'kind': 'args', 'args': ['hello', 0, 4], 'expected': 'olleh'},
            {'kind': 'args', 'args': ['ab', 0, 1], 'expected': 'ba'},
            {'kind': 'args', 'args': ['a', 0, 0], 'expected': 'a'},
        ],
    },
    104: {  # rec_sum
        'func': 'sumOfNaturalNumbers',
        'cases': [
            {'kind': 'args', 'args': [1], 'expected': 1},
            {'kind': 'args', 'args': [5], 'expected': 15},
            {'kind': 'args', 'args': [10], 'expected': 55},
        ],
    },
    105: {  # rec_intPalindrome
        'func': 'is_num_palindrome',
        'cases': [
            {'kind': 'args', 'args': [121], 'expected': True},
            {'kind': 'args', 'args': [123], 'expected': False},
            {'kind': 'args', 'args': [7], 'expected': True},
            {'kind': 'args', 'args': [1221], 'expected': True},
        ],
    },
    106: {  # rec_symmetric
        'func': 'is_symmetric',
        'cases': [
            {'kind': 'tree', 'tree': SYMMETRIC_TREE, 'expected': True},
            {'kind': 'tree', 'tree': ASYMMETRIC_TREE, 'expected': False},
            {'kind': 'tree', 'tree': None, 'expected': True},
        ],
    },
    # --- iteration ---
    200: {  # it_binSearch
        'func': 'binarySearch',
        'cases': [
            {'kind': 'args', 'args': [[2, 3, 4, 10, 40], 10], 'expected': True},
            {'kind': 'args', 'args': [[2, 3, 4, 10, 40], 5], 'expected': False},
            {'kind': 'args', 'args': [[1, 2, 3, 4, 5], 1], 'expected': True},
            {'kind': 'args', 'args': [[], 5], 'expected': False},
        ],
    },
    201: {  # it_fibonacci
        'func': 'fibonacci',
        'cases': [
            {'kind': 'args', 'args': [0], 'expected': 0},
            {'kind': 'args', 'args': [1], 'expected': 1},
            {'kind': 'args', 'args': [5], 'expected': 5},
            {'kind': 'args', 'args': [10], 'expected': 55},
        ],
    },
    202: {  # it_palindrome
        'func': 'isPalindrome',
        'cases': [
            {'kind': 'args', 'args': ['racecar'], 'expected': True},
            {'kind': 'args', 'args': ['hello'], 'expected': False},
            {'kind': 'args', 'args': ['a'], 'expected': True},
        ],
    },
    203: {  # it_reverse
        'func': 'reverseString',
        'cases': [
            {'kind': 'args', 'args': ['hello'], 'expected': 'olleh'},
            {'kind': 'args', 'args': ['ab'], 'expected': 'ba'},
        ],
    },
    204: {  # it_reverseStack (scaffold takes/returns a plain list)
        'func': 'reverseList',
        'cases': [
            {'kind': 'args', 'args': [[1, 2, 3, 4]], 'expected': [4, 3, 2, 1]},
            {'kind': 'args', 'args': [[]], 'expected': []},
            {'kind': 'args', 'args': [[1]], 'expected': [1]},
        ],
    },
    205: {  # it_sum
        'func': 'sumOfNaturalNumbers',
        'cases': [
            {'kind': 'args', 'args': [1], 'expected': 1},
            {'kind': 'args', 'args': [5], 'expected': 15},
            {'kind': 'args', 'args': [10], 'expected': 55},
        ],
    },
    206: {  # it_intPalindrome
        'func': 'is_num_palindrome',
        'cases': [
            {'kind': 'args', 'args': [121], 'expected': True},
            {'kind': 'args', 'args': [123], 'expected': False},
            {'kind': 'args', 'args': [7], 'expected': True},
            {'kind': 'args', 'args': [1221], 'expected': True},
        ],
    },
    207: {  # it_symmetric
        'func': 'is_symmetric',
        'cases': [
            {'kind': 'tree', 'tree': SYMMETRIC_TREE, 'expected': True},
            {'kind': 'tree', 'tree': ASYMMETRIC_TREE, 'expected': False},
            {'kind': 'tree', 'tree': None, 'expected': True},
        ],
    },
}


# marks where the participant's cursor was when the scan block ended - not
# something they typed. Always shows up as a single run of 2+ 'q's (with
# leading whitespace on the same line, if any), almost always exactly once
# per response
QQ_MARKER = re.compile(r'[ \t]*q{2,}')


def clean_code(code):
    """Light normalization before execution: strip the qq cursor marker,
    normalize line endings, and expand tabs to spaces (CodeMirror's
    indentUnit is 4 spaces - see reconstruct_participant_responses.py) so
    tab/space mixing doesn't produce spurious TabErrors."""
    if not isinstance(code, str):
        return code
    code = QQ_MARKER.sub('', code)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = code.expandtabs(4)
    return code


def detect_language(code):
    if not isinstance(code, str):
        return 'unknown'
    if '#include' in code or 'int main(' in code:
        return 'cpp'
    if 'def ' in code:
        return 'python'
    return 'unknown'


def run_against_test_cases(code, spec):
    payload = {'code': code, 'func_name': spec['func'], 'test_cases': spec['cases']}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(payload, f)
        input_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, RUNNER, input_path],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {'define_error': 'TIMEOUT', 'tests': []}
    finally:
        os.remove(input_path)

    if proc.returncode != 0 or not proc.stdout.strip():
        return {'define_error': f'CRASH: {proc.stderr.strip()[-500:]}', 'tests': []}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {'define_error': f'BAD_OUTPUT: {proc.stdout[:500]}', 'tests': []}


def main():
    pids = sorted(d for d in os.listdir(DATA_DIR) if os.path.isfile(f"{DATA_DIR}/{d}/{d}_task.csv"))
    rows = []
    for pid in pids:
        df = pd.read_csv(f"{DATA_DIR}/{pid}/{pid}_task.csv")
        df = df[df['stim_id'] != 'stim_id'].reset_index(drop=True)  # drop repeated header rows from concatenated blocks
        for _, row in df.iterrows():
            if row['task'] not in ('rec', 'it'):
                continue
            stim_id = int(row['stim_id'])
            spec = PROBLEMS.get(stim_id)
            if spec is None:
                continue

            code = clean_code(row['participant_output'])
            language = detect_language(code)
            record = {
                'pid': pid, 'stim_name': row['stim_name'], 'stim_id': stim_id,
                'task': row['task'], 'language': language,
            }

            if language != 'python':
                record.update(runs=False, define_error=f'skipped ({language})',
                               tests_passed=0, tests_total=len(spec['cases']))
                rows.append(record)
                continue

            result = run_against_test_cases(code, spec)
            tests = result.get('tests', [])
            runs = result.get('define_error') is None and all(t['error'] is None for t in tests)
            passed = sum(1 for t in tests if t['error'] is None and t['actual'] == t['expected'])
            record.update(
                runs=runs,
                define_error=result.get('define_error'),
                tests_passed=passed,
                tests_total=len(spec['cases']),
                details=json.dumps(tests),
            )
            rows.append(record)
            print(f"{pid} {row['stim_name']:<18} runs={runs!s:<5} passed={passed}/{len(spec['cases'])}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv('code_test_results.csv', index=False)

    n = len(out_df)
    n_runs = int(out_df['runs'].sum())
    print(f"\n{n_runs}/{n} responses ran without error.")
    print("Full results (including per-test-case actual/expected) saved to code_test_results.csv")


if __name__ == '__main__':
    main()
