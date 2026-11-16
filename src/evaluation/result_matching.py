
# ============================================================
# FILE: src/evaluation/result_matching.py
# PURPOSE: Decide whether a generated query answered correctly
#          (04_text_to_sql.ipynb, step 5)
#
# A query is correct when its result holds the same values as the
# expected query: same number of rows, and every expected row found
# among the generated rows. Column names, column order and row order
# are ignored; numbers count as equal within rounding; extra columns
# are fine. Values must sit in the same rows as in the expected result.
#
#   results_match()   generated frame, expected frame -> bool
# ============================================================

import itertools
import math


RELATIVE_TOLERANCE = 0.005

# Column pairings tried when several generated columns hold the same values.
MAX_COLUMN_CHOICES = 50


def normalize_value(value):
    """Numbers -> float, missing -> None, everything else -> text."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if hasattr(value, "item"):                      # numpy scalars
        return normalize_value(value.item())

    return str(value)


def sort_key(value):
    """Numbers first, then text, then missing values."""

    if isinstance(value, float):
        return (0, value, "")
    if isinstance(value, str):
        return (1, 0.0, value)
    return (2, 0.0, "")


def sorted_column(series):
    return sorted((normalize_value(value) for value in series), key=sort_key)


def values_equal(first, second):
    if isinstance(first, float) and isinstance(second, float):
        return math.isclose(first, second, rel_tol=RELATIVE_TOLERANCE,
                            abs_tol=0.01)
    return first == second


def columns_equal(first, second):
    return len(first) == len(second) and all(
        values_equal(a, b) for a, b in zip(first, second))


def frame_rows(frame, positions):
    """Rows of the chosen columns as tuples of normalized values."""

    columns = [[normalize_value(value) for value in frame.iloc[:, position]]
               for position in positions]

    return list(zip(*columns)) if columns else [()] * len(frame)


def rows_match(generated_rows, expected_rows):
    """
    True when every expected row pairs with a different generated row
    (row order ignored, numbers within rounding).
    """

    remaining = list(generated_rows)

    for wanted in expected_rows:
        match = next((index for index, row in enumerate(remaining)
                      if all(values_equal(a, b) for a, b in zip(row, wanted))), None)

        if match is None:
            return False

        remaining.pop(match)

    return True


def results_match(generated, expected):
    """
    True when `generated` contains every column of `expected` and the
    values sit in the same rows: (Delhi, 10), (Pune, 20) does not match
    (Delhi, 20), (Pune, 10).
    """

    if len(generated) != len(expected):
        return False

    # Step 1: generated columns that hold each expected column's values.
    available = [sorted_column(generated.iloc[:, position])
                 for position in range(generated.shape[1])]
    candidates = []

    for position in range(expected.shape[1]):
        wanted = sorted_column(expected.iloc[:, position])
        found = [index for index, column in enumerate(available)
                 if columns_equal(column, wanted)]

        if not found:
            return False

        candidates.append(found)

    # Step 2: some choice of distinct columns must also agree row by row.
    expected_rows = frame_rows(expected, range(expected.shape[1]))
    tried = 0

    for choice in itertools.product(*candidates):
        if len(set(choice)) < len(choice):
            continue

        if rows_match(frame_rows(generated, choice), expected_rows):
            return True

        tried += 1

        if tried >= MAX_COLUMN_CHOICES:
            break

    return False
