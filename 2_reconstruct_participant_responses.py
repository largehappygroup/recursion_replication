import os
import re
import ast
import glob
import pickle
import difflib
import numpy as np
import pandas as pd

# shifted alternatives of characters (s --> S, = --> +)
with open(f"misc/shift_chars.pkl", 'rb') as f:
    shift_chars = pickle.load(f)

conditions = {
    'recursion': pd.read_csv("stimuli/recursion.csv"),
    'iteration': pd.read_csv("stimuli/iteration.csv"),
    'prose'    : pd.read_csv("stimuli/prose.csv"),
    'list'     : pd.read_csv("stimuli/list.csv")
}
stimuli = pd.concat(conditions.values(), ignore_index=True)

# stimuli/*.csv were edited (typo fixes, added return-type annotations,
# reworded questions, etc.) early in data collection, so the current copy
# isn't necessarily what a given participant saw. Using correct versions
question_templates = pd.read_csv(
    "stimuli/participant_question_templates.csv",
    dtype={'participant_id': str}, keep_default_na=False
)
question_templates = {
    (row.participant_id, int(row.stim_id)): row.content
    for row in question_templates.itertuples()
}

###################################################################################
################### TEXT CLASS ####################################################
###################################################################################

INDENT_UNIT = '    '  # matches CodeMirror's indentation
DEDENT_KEYWORDS = ('else', 'elif', 'except', 'finally')
# how many keystroke rows to wait after a backspace before trusting the
# navigate-away correction trigger again
BACKSPACE_COOLDOWN_ROWS = 3

def auto_indent_for(prev_line_prefix):
    # Approximates CodeMirror's Python-mode auto-indent-on-Enter
    stripped = prev_line_prefix.lstrip(' \t')
    current_indent = prev_line_prefix[:len(prev_line_prefix) - len(stripped)]
    if stripped.rstrip().endswith(':'):
        return current_indent + INDENT_UNIT
    return current_indent

class Text:
    """
    Editable multi-line text with a single cursor.
    - text: dictionary of strings for each line that get updated by keystrokes
    - line: current line index (0-indexed)
    - col: cursor index within line (0-indexed)
    - total_lines: total number of lines in participant's response
    - shifted: boolean indicating if the last keystroke from previous timepoint was shift (i.e., need to shift current token)
    """
    def __init__(self, question_template_text: str = ""):
        self.text: dict = self.template_text(question_template_text)
        self.line: int = 0
        self.col: int = 0
        self.goal_col: int = 0 # the column a run of up/down presses is aiming for
        self.total_lines: int = len(list(self.text.keys()))
        self.shifted: bool = False
        self.auto_indent: bool = bool(re.search(r'ITERATION|RECURSION', question_template_text))

    def to_string(self) -> str:
        text = self.text.copy()
        curr_line = text[self.line]
        line_with_cursor = curr_line[0:self.col] + '|' + curr_line[self.col:]
        text[self.line] = line_with_cursor

        return '\n'.join(''.join(line) for line in text.values())

    def to_flat_string(self) -> str:
        # Full text with no cursor marker
        return '\n'.join(self.text[i] for i in range(self.total_lines))

    def template_text(self, question_template_text):
        if re.search(r'ITERATION|RECURSION', question_template_text):
            return {i: line for i,line in enumerate(question_template_text.splitlines())}
        else:
            return {0:''}
    
    # when a line is added (after pressing enter), need to shift all the following line numbers down
    def shift_line_numbers_down(self):
        for i in range(len(self.text.keys())-1, self.line, -1):
            self.text[i+1] = self.text[i]

    # when a line is deleted, need to shift all the following line numbers up
    def shift_line_numbers_up(self):
        last_idx = len(self.text.keys()) - 1
        for i in range(self.line, last_idx):
            self.text[i] = self.text[i+1]
        del self.text[last_idx]
            
    # when participant presses enter, need to increment the line count
    # and move all the text to the right of the cursor down to the following line
    def process_enter_key(self):
        # Returns the auto-indent string inserted on the new line
        right_string = (self.text[self.line])[self.col:]
        # all the text to the right is no longer on the current line
        left_string = (self.text[self.line])[:self.col]
        self.text[self.line] = left_string
        self.shift_line_numbers_down()
        self.line += 1
        self.total_lines += 1
        new_indent = auto_indent_for(left_string) if self.auto_indent else ''
        self.col = len(new_indent)
        self.text[self.line] = new_indent + right_string
        return new_indent
    
    # when left arrow keys are pressed, moving the cursor to the left, and potentially to the line(s) above if they exist
    def process_left_arrows(self, left_shift):
        for _ in range(left_shift):
            if self.col == 0:
                if self.line == 0:
                    return
                self.line -= 1
                self.col = len(self.text[self.line])
            else:
                self.col -= 1

    # when right arrow keys are pressed, moving cursor to the right, and potentially to the line(s) below if they exist
    def process_right_arrows(self, right_shift):
        for _ in range(right_shift):
            curr_line_length = len(self.text[self.line])
            if self.col == curr_line_length:
                if self.line == self.total_lines - 1:
                    return
                self.line += 1
                self.col = 0
            else:
                self.col += 1
    
    # moving cursor up if there are lines above
    def process_up_arrows(self, up_shift):
        new_line_num = self.line - up_shift

        if new_line_num < 0:
            self.line = 0
        else:
            self.line = new_line_num

        # a real editor clamps the column to the target line's length when
        # it's shorter than the line you moved from, but remembers the
        # pre-clamp goal_col so a later up/down in the same run can return
        # to it once the lines are long enough again - without this, col
        # keeps whatever got clamped on the last short/blank line passed
        # through, and every keystroke applied from here on (especially
        # backspace, whose col==0 check decides whether to join lines)
        # silently misfires
        self.col = min(self.goal_col, len(self.text[self.line]))

    # moving cursor down if there are lines below
    def process_down_arrows(self, down_shift):
        new_line_num = self.line + down_shift

        if new_line_num > (self.total_lines) - 1:
            self.line = (self.total_lines) - 1
        else:
            self.line = new_line_num

        # see process_up_arrows - same out-of-bounds-col issue
        self.col = min(self.goal_col, len(self.text[self.line]))
    
    # processing delete key
    def process_one_backspace(self):
        # Returns the character deleted if the edit needs to be replayed/undone
        # at the start of a line, backspace joins this line onto the end of the previous one
        if self.col == 0:
            if self.line == 0:
                return ''
            prev_line_length = len(self.text[self.line-1])
            self.text[self.line-1] = self.text[self.line-1] + self.text[self.line]
            self.shift_line_numbers_up()
            self.line -= 1
            self.col = prev_line_length
            self.total_lines = max(1, self.total_lines - 1)
            return '\n'
        else:
            # deleting character to the left of the cursor
            left_string = (self.text[self.line])[:self.col-1]
            deleted_char = (self.text[self.line])[self.col-1]

            # keeping track of characters to the right of cursor
            right_string = (self.text[self.line])[self.col:]

            # concatenating new string
            self.text[self.line] = left_string + right_string
            self.col -= 1
            return deleted_char

    # decrementing text, line numbers, and cursor position for each occurrence of backspace key
    def process_multiple_backspaces(self, del_count):
        # Returns the full deleted substring in original reading order 
        deleted = ''
        for _ in range(del_count):
            deleted = self.process_one_backspace() + deleted
        return deleted

    # replacing a token with shifted version
    def shift_token(self, shifted_token):
        result = self.append_token(shifted_token)
        self.shifted = False
        return result

    # if it's just a normal token, append it to current cursor position
    def append_token(self, token):
        """
        Returns the indent text removed by an electric dedent triggered by
        this token ('' if none) - see maybe_dedent_electric_keyword.
        """
        left_string = (self.text[self.line])[:self.col] + token
        right_string = (self.text[self.line])[self.col:]
        new_string = left_string + right_string
        self.text[self.line] = new_string
        self.col += len(token)
        return self.maybe_dedent_electric_keyword()

    def maybe_dedent_electric_keyword(self):
        """
        Mimics CodeMirror's "electric" dedent: the moment the line being
        typed (ignoring its leading whitespace, and only counting what's
        typed up to the cursor) becomes exactly one of Python's
        dedent-triggering keywords, the editor snaps that line's indent
        back by one level.

        Returns the indent text removed from the front of the line ('' if
        no dedent happened) - like an auto-inserted indent, this is
        synthesized, not a literal keystroke (or backspace), so it has to
        be recorded separately for anything that needs to replay/undo it.
        """
        if not self.auto_indent:
            return ''

        line = self.text[self.line]
        typed_so_far = line[:self.col]
        stripped = typed_so_far.lstrip(' \t')
        if stripped not in DEDENT_KEYWORDS:
            return ''

        indent_len = len(typed_so_far) - len(stripped)
        if indent_len < len(INDENT_UNIT):
            return ''

        new_indent = ' ' * (indent_len - len(INDENT_UNIT))
        tail = line[self.col:]
        self.text[self.line] = new_indent + stripped + tail
        self.col = len(new_indent) + len(stripped)
        return INDENT_UNIT


##############################################################################
####################### FUNCTIONS ############################################
##############################################################################

def update_text_and_cursor_position(new_text, answer):
    # Returns a list of (token_index, operation, text) edit events that occurred
    # while processing new_text
    edits = []

    for i,token in enumerate(new_text):
        one_bs_match   = re.search('<K:BS>', token)
        many_bs_match  = re.search(r'<K:BS x=([0-9]+)>', token)
        ctrl_match  = re.search('<K:CTRL', token)
        shift_match = re.search('<K:S', token)

        left_match  = re.search(r'(<K:L>)|(<K:L x=([0-9]+))', token)
        right_match = re.search(r'(<K:R>)|(<K:R x=([0-9]+))', token)
        up_match    = re.search(r'(<K:U>)|(<K:U x=([0-9]+))', token)
        down_match  = re.search(r'(<K:D>)|(<K:D x=([0-9]+))', token)

        if re.search(r'\t', token):
            dedented = answer.append_token('    ')
            if dedented:
                edits.append((i, 'synthesized', dedented))
            answer.goal_col = answer.col
            continue

        if re.search(r'\r', token):
            inserted_indent = answer.process_enter_key()
            if inserted_indent:
                edits.append((i, 'synthesized', inserted_indent))
            answer.goal_col = answer.col
            continue

        if left_match:
            left_shift  = 1 if not left_match.group(3) else int(left_match.group(3))
            answer.process_left_arrows(left_shift)
            answer.goal_col = answer.col
            continue

        elif right_match:
            right_shift = 1 if not right_match.group(3) else int(right_match.group(3))
            answer.process_right_arrows(right_shift)
            answer.goal_col = answer.col
            continue

        elif up_match:
            up_shift = 1 if not up_match.group(3) else int(up_match.group(3))
            answer.process_up_arrows(up_shift)
            continue

        elif down_match:
            down_shift  = 1 if not down_match.group(3) else int(down_match.group(3))
            answer.process_down_arrows(down_shift)
            continue

        if ctrl_match or shift_match:
            if i == len(new_text)-1 and shift_match: # if the last token is a shift key
                answer.shifted = True
            continue

        if one_bs_match:
            deleted = answer.process_one_backspace()
            if deleted:
                edits.append((i, 'delete', deleted))

        elif many_bs_match:
            del_count = int(many_bs_match.group(1))
            deleted = answer.process_multiple_backspaces(del_count)
            if deleted:
                edits.append((i, 'delete', deleted))

        elif answer.shifted:
            try:
                shifted_token = shift_chars[token]
            except:
                answer.text[answer.line] += token
                answer.col += len(token)
                continue
            dedented = answer.shift_token(shifted_token)
            if dedented:
                edits.append((i, 'synthesized', dedented))
        else:
            dedented = answer.append_token(token)
            if dedented:
                edits.append((i, 'synthesized', dedented))

        answer.goal_col = answer.col

    return edits

def combine_shift_sequences(vol_text):
    combined_text = []
    
    shifted = False
    for i,t in enumerate(vol_text):
        if shifted:
            shifted = False
            continue
        if re.search("<K:S", t):
            if i < (len(vol_text)-1):
                next_key = vol_text[i+1]
                try:
                    shifted_char = shift_chars[next_key]
                except:
                    # print(f"No entry for {next_key}, {ascii(next_key)}")
                    continue
                combined_text.append(shifted_char)
                shifted = True
            else:
                combined_text.append(t)       
        else:
            combined_text.append(t)
            
    return combined_text

def get_question_text(question_num, person=None):
    if person is not None and (str(person), question_num) in question_templates:
        return question_templates[(str(person), question_num)]
    question_idx = np.where(stimuli['stim_id'] == question_num)
    question_text = (list(stimuli.loc[question_idx, 'content']))[0]
    return question_text


# Contextualizing the keystrokes at each timepoint with corresponding question context 
# Also adding tags for <PRE><SUF><MID> 
def format_keystrokes(vol, keystroke_log):
    prompts = {
        'recursion' : 'Please fill in the scaffolding for this question using a recursive implementation in Python, where active flow involves making recursive calls that build up the recursive stack, and passive flow is the work on the way down as the calls finish executing.',
        'iteration' : 'Please fill in the scaffolding for this question using an iterative implementation in Python.',
        'prose'     : 'Please use right-branching phrases following this same phrase structure to add 10 more examples for the following sentence.',
        'list'      : ''
    }
    vol_text   = keystroke_log[vol]['full_text']
    stim_id    = keystroke_log[vol]['stim_id']
    keystrokes = ''.join(keystroke_log[vol]['keystrokes'])

    if not stim_id:
        return None
    if stim_id < 200:
        task = 'recursion'
    elif stim_id >= 200 and stim_id < 300:
        task = 'iteration'
    elif stim_id >= 300 and stim_id < 400:
        task = 'prose'
    elif stim_id >= 400:
        task = 'list'

    question_text = list(stimuli.loc[np.where(stimuli['stim_id'] == stim_id), 'content'])[0]

    if not vol_text:
        return f"<PRE>{prompts[task]+question_text}<SUF><MID>{keystrokes}"

    lines = vol_text.split('\n')
    cursor_row = keystroke_log[vol]['row']
    cursor_col = keystroke_log[vol]['col']

    offset = sum(len(line) + 1 for line in lines[:cursor_row]) + cursor_col
    prefix = vol_text[:offset]
    suffix = vol_text[offset:]

    if task in ['prose', 'list']:
        return f"<PRE>{prompts[task]+question_text+prefix}<SUF>{suffix}<MID>{keystrokes}"
    else:
        return f"<PRE>{prompts[task]+prefix}<SUF>{suffix}<MID>{keystrokes}"


# Loads the ground-truth final answer and the end_time they finished it
def load_task_metadata(person):
    oracle_path = f"data/{person}/{person}_task.csv"
    try:
        df = pd.read_csv(oracle_path)
    except FileNotFoundError:
        return {}, {}

    df = df[df['stim_id'] != 'stim_id']
    answers = {}
    end_times = {}
    for _, row in df.iterrows():
        stim_id = int(row['stim_id'])
        # normalize CRLF from the csv down to the \n the Text class uses
        answers[stim_id] = str(row['participant_output']).replace('\r\n', '\n')
        end_times[stim_id] = pd.Timestamp(row['end_time'])
    return answers, end_times

# Rewrites answer's text/line/col from new_full_text (the corrected text) and cursor_offset.
def _apply_correction(answer, new_full_text, cursor_offset):
    new_lines = new_full_text.split('\n')
    answer.text = {i: line for i, line in enumerate(new_lines)}
    answer.total_lines = len(new_lines)

    running = 0
    for i, line in enumerate(new_lines):
        if running + len(line) >= cursor_offset:
            answer.line = i
            answer.col = cursor_offset - running
            answer.goal_col = answer.col
            return
        running += len(line) + 1

    # cursor_offset landed past the end of new_full_text 
    # - clamp to the last valid position instead of silently 
    # leaving answer.line/col at whatever they were before this call. 
    answer.line = len(new_lines) - 1
    answer.col = len(new_lines[-1])
    answer.goal_col = answer.col

# Finds where probe matches something in oracle text
# min_match requires that probe matches verbatim, not just a substring of it 
# Among matches, this picks whichever is closest to the current cursor
# declines if there's a rival match
# Returns the resulting target_offset, or None if no confident match exists.
def _find_jump_target(probe, oracle_text, cursor_offset, min_match, ambiguity_margin):
    if len(probe) < min_match:
        return None

    matcher = difflib.SequenceMatcher(None, probe, oracle_text, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= min_match]
    if not blocks:
        return None

    tail_blocks = [b for b in blocks if b.a + b.size == len(probe) and b.b + b.size > cursor_offset]
    if not tail_blocks:
        return None

    def distance_from_cursor(b):
        return abs((b.b + b.size) - cursor_offset)

    best = min(tail_blocks, key=lambda b: (distance_from_cursor(b), -b.size))

    runner_up_distance = min(
        (distance_from_cursor(b) for b in tail_blocks if b is not best),
        default=None
    )
    if runner_up_distance is not None and runner_up_distance - distance_from_cursor(best) <= ambiguity_margin:
        return None

    return best.b + best.size

def count_keystrokes(keys_list):
    sum = 0
    for key in keys_list:
        if key == '':
            continue
        elif re.search('[0-9]+>', key) and not re.search(r'^<K:S|^<K:CTRL', key):
            num_presses = (re.search(r'([0-9]+)|>', key))[0]
            sum += int(num_presses)
        else:
            sum += 1
    return sum

def realign_completed_lines(answer, oracle_text, completed_line_indices, min_match=15, ambiguity_margin=3,
                             recent_chars=20, wide_min_match=20, local_min_ratio=0.6):
    """
    Only corrects up through the last completed line - whatever comes after 
    Returns True if a correction was made, False otherwise.
    Triggers once the current line is stable (after participant navigates away)
    
    completed_line_indices: one or more consecutive closed line indices, oldest first. 
    Passing more than one line joins their content into a single probe
    This avoids issues where a single line can match another similar line elsewhere in the code
    
    Tries three checks:
    1. Tight jump - the forward-only, closest-match, ambiguity-guarded
       search from _find_jump_target, using the completed line's own
       content as the probe.
    2. Wide jump - if that finds nothing confident (e.g. the line's too
       short), fall back to the last recent_chars characters ending at the
       completed line.
    3. Local repair - both jump searches require the *entire* probe to
       match somewhere verbatim. Another failure mode is that the text can get
       garbled. This just replaces it with oracle text based on the cursor position.

    """
    if not completed_line_indices or any(idx not in answer.text for idx in completed_line_indices):
        return False

    last_line_idx = completed_line_indices[-1]
    lines = [answer.text[i] for i in range(answer.total_lines)]
    line_end_offset = sum(len(l) + 1 for l in lines[:last_line_idx]) + len(lines[last_line_idx])
    full_text = '\n'.join(lines)
    tight_probe = '\n'.join(lines[idx] for idx in completed_line_indices)

    if tight_probe and oracle_text[:line_end_offset].endswith(tight_probe):
        return False  # already aligned

    target_offset = None
    if len(tight_probe) >= min_match:
        target_offset = _find_jump_target(tight_probe, oracle_text, line_end_offset, min_match, ambiguity_margin)

    if target_offset is None:
        wide_probe = full_text[:line_end_offset][-recent_chars:]
        target_offset = _find_jump_target(wide_probe, oracle_text, line_end_offset, wide_min_match, ambiguity_margin)

    if target_offset is None and tight_probe:
        local_start = line_end_offset - len(tight_probe)
        local_target = oracle_text[local_start:line_end_offset]
        if local_target and local_target != tight_probe:
            ratio = difflib.SequenceMatcher(None, tight_probe, local_target, autojunk=False).ratio()
            if ratio >= local_min_ratio:
                target_offset = line_end_offset

    if target_offset is None:
        return False

    # oracle_text[:target_offset] is a character-offset slice, not a
    # line-aware one - nothing above guarantees it lands on the same
    # line-count boundary
    correct_prefix = oracle_text[:target_offset]

    if correct_prefix.count('\n') != last_line_idx:
        return False

    tail = full_text[line_end_offset:]

    cursor_offset_before = sum(len(l) + 1 for l in lines[:answer.line]) + answer.col
    new_cursor_offset = cursor_offset_before + (len(correct_prefix) - line_end_offset)
    _apply_correction(answer, correct_prefix + tail, new_cursor_offset)
    return True

# Companion to realign_completed_lines, which only fixes the live `answer` object
# This rewrites the buffer that hasn't been committed yet with the corrected text
def apply_line_correction_to_buffer(line_buffer, completed_line_idx, corrected_line_text):

    for _, entry in line_buffer:
        if entry['row'] != completed_line_idx:
            continue
        reveal_len = min(entry['col'], len(corrected_line_text))
        lines = entry['full_text'].split('\n')
        lines[completed_line_idx] = corrected_line_text[:reveal_len]
        entry['full_text'] = '\n'.join(lines)
        entry['col'] = reveal_len

def lookahead_correct_lines(line_buffer, answer, oracle_text, completed_line_indices, min_match=15, ambiguity_margin=3):
    last_line_idx = completed_line_indices[-1]
    snapshot = (dict(answer.text), answer.line, answer.col, answer.goal_col, answer.total_lines)
    line_before_call = answer.line

    if not realign_completed_lines(answer, oracle_text, completed_line_indices, min_match=min_match, ambiguity_margin=ambiguity_margin):
        return False

    if answer.line != line_before_call:
        answer.text, answer.line, answer.col, answer.goal_col, answer.total_lines = snapshot
        return False

    for idx in completed_line_indices:
        apply_line_correction_to_buffer(line_buffer, idx, answer.text[idx])

    # every buffered entry past the corrected lines sits in the untouched
    # tail, so its own row/col are still valid
    new_prefix = '\n'.join(answer.text[idx] for idx in range(last_line_idx + 1))
    for _, entry in line_buffer:
        if entry['row'] > last_line_idx:
            old_lines = entry['full_text'].split('\n')
            entry['full_text'] = new_prefix + '\n' + '\n'.join(old_lines[last_line_idx + 1:])
            entry['total_lines'] = answer.total_lines

    return True

def entry_to_string(entry):
    """Debug-print helper: entry's full_text with a cursor marker inserted, matching Text.to_string()'s format."""
    lines = entry['full_text'].split('\n')
    r, c = entry['row'], entry['col']
    lines[r] = lines[r][:c] + '|' + lines[r][c:]
    return '\n'.join(lines)

def num_keystrokes_pressed_per_volume(keystroke_dict):

    # np.array(list(num_keystrokes.values()))
    num_keystrokes_pressed = np.array([keystroke_dict[vol]['num_keystrokes'] for vol in list(keystroke_dict.keys())])
    return num_keystrokes_pressed

def isolate_new_keystrokes(keystroke_dict):
    new_keystrokes = {vol: keystroke_dict[vol]['keystrokes'] for vol in list(keystroke_dict.keys())}
    return new_keystrokes

def process_keystrokes(vol_keystroke_file, person, align_with_oracle, save_log=True, debug_print=False):

    vol_keystroke_df = pd.read_csv(vol_keystroke_file)

    # per-volume log: keys are volume numbers 
    # values include cursor position and anything the forward pass deleted or
    # synthesized that isn't recoverable from the raw keystroke tokens alone 
    keystroke_log = {}
    # this question's own template text
    template_texts_by_stim = {}

    def commit(vi, entry):
        keystroke_log[vi] = entry
        # if debug_print:
        #     print(f"UPDATE Total - lines: {entry['total_lines']}, row: {entry['row']}, col: {entry['col']}, text: {entry_to_string(entry)}, {entry['keystrokes']}")

    oracle_answers, stim_end_times = load_task_metadata(person)

    prev_question = -1
    question_num = None
    answer = Text()
    # rows since the participant last pressed backspace - used to hold off
    # the navigate-away trigger below while a revision is likely still in progress
    rows_since_backspace = BACKSPACE_COOLDOWN_ROWS

    # correction holds back entries for the line currently being typed them to keystroke_log immediately - 
    # line_buffer is that holding pen, as (vol_num, entry) pairs in volume order; 
    line_buffer = []
    buffering_this_question = False
    # the most recently closed line that hasn't been decided yet - lookahead
    # holds a line's correction decision open for one extra line-boundary so
    # it can try pairing it with whatever comes right after
    pending_line_idx = None
    # the open_line_idx (see the navigate-away branch below) the backspace
    # cooldown has already cleared for once
    nav_gate_cleared_for = None

    for i,row in vol_keystroke_df.iterrows():
        try:
            curr_question = int(row['stim_id'])
        except (TypeError, ValueError):
            # NaN stim_id: rest, either before the first question or
            # between two questions - not part of any answer
            curr_question = None

        # line number and cursor_index should reset with each question
        if curr_question is not None and curr_question != prev_question:
            # a pending line from the previous question never got a
            # second line to pair with - decide it alone (see the
            # lookahead branch below) before flushing, its last chance
            if pending_line_idx is not None and buffering_this_question:
                lookahead_correct_lines(line_buffer, answer, oracle_answers.get(prev_question, ''), [pending_line_idx])
            pending_line_idx = None
            nav_gate_cleared_for = None

            # flush whatever's left of the previous question's still-open
            # line window - no Enter ever closed it, so there's no anchor
            # to check it against; commit it as-is, uncorrected
            for vi, e in line_buffer:
                commit(vi, e)
            line_buffer = []

            prev_question = curr_question
            question_num = curr_question

            # creating new text object to contain participant's response to current question
            template_text = get_question_text(question_num, person=person)
            template_texts_by_stim[question_num] = template_text
            answer = Text(template_text)
            rows_since_backspace = BACKSPACE_COOLDOWN_ROWS  # no revision-in-progress carries over between questions
            buffering_this_question = align_with_oracle and answer.auto_indent

        # rest volume
        if curr_question is None:
            commit(i, {
                'stim_id': None,
                'row': 0,
                'col': 0,
                'total_lines': 1,
                'keystrokes': [],
                'num_keystrokes': 0,
                'full_text': '',
            })
            continue

        vol_text = ast.literal_eval(row['keystrokes'])

        if len(vol_text) == 0:
            idle_entry = {
                'stim_id': question_num,
                'row': answer.line,
                'col': answer.col,
                'total_lines': answer.total_lines,
                'keystrokes': [],
                'num_keystrokes':0,
                'full_text': answer.to_flat_string(),
            }
            if buffering_this_question:
                line_buffer.append((i, idle_entry))
            else:
                commit(i, idle_entry)
            continue
        
        shift_combined = combine_shift_sequences(vol_text)

        entry = {
            'stim_id': question_num,
            'row': answer.line,
            'col': answer.col,
            'total_lines': answer.total_lines,
            'keystrokes': shift_combined,
            'num_keystrokes': count_keystrokes(shift_combined),
            'full_text': answer.to_flat_string(),
        }

        edits = update_text_and_cursor_position(shift_combined, answer)

        if buffering_this_question:
            line_buffer.append((i, entry))
        else:
            commit(i, entry)

        if any(re.search('<K:BS', token) for token in shift_combined):
            rows_since_backspace = 0
        else:
            rows_since_backspace += 1

        if align_with_oracle and answer.auto_indent and question_num in oracle_answers:
            if '\r' in shift_combined:
                completed_line_idx = answer.line - 1

                if pending_line_idx is not None and pending_line_idx == completed_line_idx - 1:
                    joint = [pending_line_idx, completed_line_idx]

                    if lookahead_correct_lines(line_buffer, answer, oracle_answers[question_num], joint):
                        print(f"LOOKAHEAD CORRECTION (joint) at row {i} - snapped completed lines {pending_line_idx}-{completed_line_idx} to oracle: lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}")
                    elif lookahead_correct_lines(line_buffer, answer, oracle_answers[question_num], [pending_line_idx]):
                        print(f"LOOKAHEAD CORRECTION at row {i} - snapped completed line {pending_line_idx} to oracle: lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}")

                elif pending_line_idx is not None:
                    # non-consecutive (e.g. two Enters landed in the
                    # same volume) - nothing to pair it with, decide
                    # it alone rather than let it wait indefinitely
                    if lookahead_correct_lines(line_buffer, answer, oracle_answers[question_num], [pending_line_idx]):
                        print(f"LOOKAHEAD CORRECTION at row {i} - snapped completed line {pending_line_idx} to oracle: lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}")

                # pending_line_idx (if any) is now fully decided -
                # flush every buffered entry up to and including it;
                # the line that just closed becomes the new pending line
                if pending_line_idx is not None:
                    for vi, e in line_buffer:
                        if e['row'] <= pending_line_idx:
                            commit(vi, e)
                    line_buffer[:] = [(vi, e) for vi, e in line_buffer if e['row'] > pending_line_idx]
                pending_line_idx = completed_line_idx

            elif answer.line - (pending_line_idx + 1 if pending_line_idx is not None else 0) >= 2:
                # the cursor has navigated (e.g. arrow keys) at least 2
                # lines past the oldest still-open line 
                open_line_idx = pending_line_idx + 1 if pending_line_idx is not None else 0
                gate_cleared = nav_gate_cleared_for == open_line_idx
                if gate_cleared or rows_since_backspace >= BACKSPACE_COOLDOWN_ROWS:
                    nav_gate_cleared_for = open_line_idx
                    if lookahead_correct_lines(line_buffer, answer, oracle_answers[question_num], [open_line_idx]):
                        print(f"LOOKAHEAD CORRECTION (nav) at row {i} - snapped departed line {open_line_idx} to oracle: lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}")

        if debug_print:
            print(answer.to_string(), shift_combined, f"ROW: {answer.line} | COL: {answer.col}")

    # a pending line from the final question never got a second line to
    # pair with - decide it alone before flushing, its last chance
    if pending_line_idx is not None and buffering_this_question:
        lookahead_correct_lines(line_buffer, answer, oracle_answers.get(question_num, ''), [pending_line_idx])
    pending_line_idx = None

    # flush whatever's left of the final question's still-open line window
    for vi, e in line_buffer:
        commit(vi, e)
    line_buffer = []

    # per-volume log 
    if save_log:
        scan_block = os.path.basename(vol_keystroke_file).removesuffix('_keystroke_df.csv')
        num_keystroke_regressor = num_keystrokes_pressed_per_volume(keystroke_log)
        regressor_outpath = f"midprocess/{person}/{scan_block}-num_keystrokes_regressor.pkl"
        with open(regressor_outpath, 'wb') as f:
            pickle.dump(num_keystroke_regressor, f)

        new_keystrokes = isolate_new_keystrokes(keystroke_log)
        new_keystroke_outpath = f"midprocess/{person}/{scan_block}-new_keystrokes.pkl"
        with open(new_keystroke_outpath, 'wb') as f:
            pickle.dump(new_keystrokes, f)

        keystroke_log = {vol: format_keystrokes(vol, keystroke_log) for vol in list(keystroke_log.keys())}
        keystroke_log = {vol: keys for vol,keys in keystroke_log.items() if keys}
        
        log_outpath = f"midprocess/{person}/{scan_block}_keystroke_log.pkl"
        
        # for k,v in keystroke_log.items():
            # print(k,v)
        with open(log_outpath, 'wb') as f:
            pickle.dump(keystroke_log, f)


    return keystroke_log

def process_participant(person, participant_path, align_with_oracle, save_log=True, debug_print=False):
    keystroke_files = glob.glob(f"{participant_path}/*_keystroke_df.csv")
    for file in keystroke_files:
        process_keystrokes(file, person, align_with_oracle=align_with_oracle, save_log=save_log, debug_print=debug_print)
        # break

def main(align_with_oracle=True, save_log=True, debug_print=False):
    datapath = f"midprocess"
    participants = sorted(os.listdir(datapath))
    # participants = ['105', '110', '114', '120', '130']

    for person in participants:
        print(person)
        participant_path = f"{datapath}/{person}"
        process_participant(person, participant_path, align_with_oracle=align_with_oracle,
                             save_log=save_log, debug_print=debug_print)

        # break

if __name__ == "__main__":
    main()
