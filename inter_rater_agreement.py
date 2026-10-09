import pickle
import warnings
import numpy as np
import krippendorff

# THIS SCRIPT COMPUTES KRIPPENDORFF'S ALPHA BETWEEN THE THREE RATERS (A, B, C) WHO SCORED
# PARTICIPANTS' TASK RESPONSES (SEE answer_ratings/rate_answers.py), AND CONSOLIDATES THEIR
# THREE RATINGS INTO A SINGLE VALUE PER QUESTION/CRITERION.
#
# EACH RATED QUESTION KEY LOOKS LIKE "<task>_<stim_name>_<stim_id>" WHERE <task> IS ONE OF
# rec/it/pro/lis, AND MAPS TO A 4-ELEMENT LIST: [correct, efficient, complete, readable].
#   - correct:   for recursion/iteration, this is tests_passed/tests_total, computed automatically
#                from code_test_results.csv (NOT a subjective rating, so it's excluded from the
#                alpha and valence checks below - it's just carried through as-is).
#                for prose/list, this IS a 1-5 rating ("the answer is correct").
#   - efficient: only rated for recursion/iteration ("uses an efficient approach"); always blank
#                ('') for prose/list.
#   - complete:  1-5 rating, collected for every condition.
#   - readable:  1-5 rating, collected for every condition.
#
# CONSOLIDATION: for every rated (question, criterion), each rater's 1-5 score is bucketed into a
# valence - negative (<3), neutral (==3), or positive (>3). If all three raters land in the same
# valence bucket, the median of their three scores is taken as the consolidated value. If they
# don't (e.g. one neutral + two positive, or one of each), or if any rater's value is missing,
# non-numeric, or outside the 1-5 scale, the item is flagged for manual review instead.

RATERS = ['A', 'B', 'C']
RATINGS_DIR = "answer_ratings"
CONSOLIDATED_PATH = f"{RATINGS_DIR}/Consolidated_Reviews.pkl"

FIELDS = ['correct', 'efficient', 'complete', 'readable']

# how each field should be treated, per task-name prefix:
#   'rating'   - a subjective 1-5 score from each rater -> valence-checked and consolidated
#   'computed' - an objective value computed once (not rated independently) -> carried through as-is
#   'n/a'      - not collected for this condition
PREFIX_FIELD_TYPES = {
    'rec': {'correct': 'computed', 'efficient': 'rating', 'complete': 'rating', 'readable': 'rating'},
    'it':  {'correct': 'computed', 'efficient': 'rating', 'complete': 'rating', 'readable': 'rating'},
    'pro': {'correct': 'rating', 'efficient': 'n/a', 'complete': 'rating', 'readable': 'rating'},
    'lis': {'correct': 'rating', 'efficient': 'n/a', 'complete': 'rating', 'readable': 'rating'},
}

# which fields to compute inter-rater alpha for, grouped by condition
CONDITION_GROUPS = {
    'recursion/iteration': {
        'prefixes': {'rec', 'it'},
        'fields': ['efficient', 'complete', 'readable'],
    },
    'prose/list': {
        'prefixes': {'pro', 'lis'},
        'fields': ['correct', 'complete', 'readable'],
    },
}


def load_ratings():
    ratings = {}
    for rater in RATERS:
        with open(f"{RATINGS_DIR}/Task_Response_Ratings_{rater}.pkl", "rb") as f:
            ratings[rater] = pickle.load(f)
    return ratings


# parses a raw 1-5 rating value into (numeric_value, issue), where issue is None if the value
# is a valid rating, or a short string describing the problem otherwise
def parse_rating(value):
    if value == '':
        return np.nan, 'missing'
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return np.nan, 'non_numeric'
    if not (1 <= numeric_value <= 5):
        return numeric_value, 'out_of_range'
    return numeric_value, None


def valence(value):
    if value > 3:
        return 'positive'
    if value < 3:
        return 'negative'
    return 'neutral'


# builds a (n_raters x n_items) reliability matrix for one field within one condition group
def build_reliability_matrix(ratings, prefixes, field, data_warnings):
    field_idx = FIELDS.index(field)

    # items are the union of (pid, key) pairs across all raters that belong to this condition group
    items = set()
    for rater in RATERS:
        for pid, questions in ratings[rater].items():
            for key in questions:
                if key.split('_')[0] in prefixes:
                    items.add((pid, key))
    items = sorted(items)

    matrix = np.full((len(RATERS), len(items)), np.nan)
    for row, rater in enumerate(RATERS):
        for col, (pid, key) in enumerate(items):
            value = ratings[rater].get(pid, {}).get(key)
            if value is None:
                continue
            numeric_value, issue = parse_rating(value[field_idx])
            if issue:
                data_warnings.append(f"  rater {rater}, pid {pid}, {key} ({field}): {issue} value {value[field_idx]!r}")
            matrix[row, col] = numeric_value

    return matrix


def report_alpha(ratings):
    data_warnings = []

    print("Krippendorff's alpha (ordinal) between raters A, B, C\n")

    for group_name, group in CONDITION_GROUPS.items():
        print(f"=== {group_name} ===")
        for field in group['fields']:
            matrix = build_reliability_matrix(ratings, group['prefixes'], field, data_warnings)
            with warnings.catch_warnings():
                # krippendorff warns when a unit has fewer than two raters; those units still
                # contribute no information to alpha, so the warning is safe to suppress here
                warnings.simplefilter("ignore")
                alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
            n_items = matrix.shape[1]
            n_rated = np.sum(~np.isnan(matrix).all(axis=0))
            print(f"  {field:<10s} alpha = {alpha:.4f}   (n items = {n_items}, n rated by >=1 rater = {n_rated})")
        print()

    if data_warnings:
        print("Data issues found while building the reliability matrices:")
        for warning in data_warnings:
            print(warning)
        print()


# consolidates the three raters' scores for one 'computed' field (e.g. code test correctness),
# which should be identical across raters since it isn't an independent judgment
def consolidate_computed(raw_by_rater):
    numeric_by_rater = {r: (float(v) if v not in (None, '') else np.nan) for r, v in raw_by_rater.items()}
    present = [v for v in numeric_by_rater.values() if not np.isnan(v)]
    mismatched = len(set(present)) > 1
    return {
        'raters': raw_by_rater,
        'needs_review': mismatched,
        'reason': 'rater_mismatch' if mismatched else None,
        'value': present[0] if present and not mismatched else None,
    }


# consolidates the three raters' scores for one 'rating' field, flagging a valence difference or
# any data issue (missing/non-numeric/out-of-range value) for manual review, and otherwise taking
# the median of the three scores
def consolidate_rating(raw_by_rater):
    parsed = {r: parse_rating(v) for r, v in raw_by_rater.items()}
    issues = {r: issue for r, (_, issue) in parsed.items() if issue}

    if issues:
        return {
            'raters': raw_by_rater,
            'needs_review': True,
            'reason': 'data_issue',
            'issues': issues,
            'value': None,
        }

    numeric_by_rater = {r: value for r, (value, _) in parsed.items()}
    valence_by_rater = {r: valence(value) for r, value in numeric_by_rater.items()}
    if len(set(valence_by_rater.values())) > 1:
        return {
            'raters': raw_by_rater,
            'needs_review': True,
            'reason': 'valence_difference',
            'valence': valence_by_rater,
            'value': None,
        }

    return {
        'raters': raw_by_rater,
        'needs_review': False,
        'reason': None,
        'value': float(np.median(list(numeric_by_rater.values()))),
    }


def consolidate_ratings(ratings):
    consolidated = {}
    review_items = []

    all_pids = sorted(set().union(*(set(ratings[r].keys()) for r in RATERS)), key=int)
    for pid in all_pids:
        consolidated[pid] = {}
        all_keys = sorted(set().union(*(set(ratings[r].get(pid, {}).keys()) for r in RATERS)))

        for key in all_keys:
            prefix = key.split('_')[0]
            field_types = PREFIX_FIELD_TYPES[prefix]
            consolidated[pid][key] = {}

            for field, field_type in field_types.items():
                if field_type == 'n/a':
                    continue

                field_idx = FIELDS.index(field)
                raw_by_rater = {r: ratings[r].get(pid, {}).get(key, [None] * 4)[field_idx] for r in RATERS}

                if field_type == 'computed':
                    entry = consolidate_computed(raw_by_rater)
                else:
                    entry = consolidate_rating(raw_by_rater)

                consolidated[pid][key][field] = entry
                if entry['needs_review']:
                    review_items.append({'pid': pid, 'key': key, 'field': field, **entry})

    return consolidated, review_items


REVIEW_ITEMS_PATH = f"{RATINGS_DIR}/items_needing_review.txt"


def report_review_items(review_items):
    counts = {}
    for item in review_items:
        counts[item['reason']] = counts.get(item['reason'], 0) + 1

    print(f"{len(review_items)} question/criterion combinations need manual review:")
    for reason, count in counts.items():
        print(f"  {reason:<18s} {count}")

    with open(REVIEW_ITEMS_PATH, "w") as f:
        for item in review_items:
            detail = item['valence'] if item['reason'] == 'valence_difference' else item.get('issues', item['raters'])
            f.write(f"pid {item['pid']:<4s} {item['key']:<22s} {item['field']:<10s} reason={item['reason']:<18s} "
                    f"raters={item['raters']}  {detail}\n")
    print(f"Full list written to {REVIEW_ITEMS_PATH}\n")


def main():
    ratings = load_ratings()

    report_alpha(ratings)

    consolidated, review_items = consolidate_ratings(ratings)
    report_review_items(review_items)

    # with open(CONSOLIDATED_PATH, "wb") as f:
    #     pickle.dump(consolidated, f)
    # print(f"Saved consolidated ratings to {CONSOLIDATED_PATH}")


if __name__ == "__main__":
    main()
