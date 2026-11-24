# ============================================================
# FILE: tests/test_cleaning_review.py
# PURPOSE: Mixed date formats, list columns, the review step's choices,
#          long-summary check and the "top 5 rows" rule. No model.
# ============================================================

import pandas as pd
import pytest

from src.helpers.answer_summary import build_caveats, summary_problem
from src.helpers.cleaning_review import build_review, suggested_choices, validate_choices
from src.helpers.data_cleaner import clean_table, split_list_columns, try_date, new_report
from src.helpers.message_intent import classify_message, row_preview_request


def grid(columns, rows):
    """A table as the file loader returns it: header row first."""

    return {"grid": pd.DataFrame([columns] + rows, dtype=object), "sheet": None,
            "source_info": {}, "header_known": True}


# ------------------------------------------------------------
# Dates in many formats
# ------------------------------------------------------------

def test_mixed_date_formats_are_read():
    odd = ["09 21 1972", "23 -07-2008", "22 Feb 04", "10-29-99", "23rd December of 1966",
           "21-11-46", "The 6th of marzo, year 1951"]
    values = pd.Series(odd + [f"2000-01-{day:02d}" for day in range(1, 26)])

    result = try_date(values)
    read = dict(result["guessed"])

    assert result["kind"] == "date" and result["failed"].empty
    assert read["23rd December of 1966"] == "1966-12-23"
    assert read["The 6th of marzo, year 1951"] == "1951-03-06"
    assert read["21-11-46"] == "1946-11-21"          # never a future year
    assert read["10-29-99"] == "1999-10-29"


def test_mostly_text_is_not_turned_into_dates():
    values = pd.Series(["apple", "pear", "2001-01-01", "plum"] * 5)

    assert try_date(values) is None


# ------------------------------------------------------------
# List columns
# ------------------------------------------------------------

TAGS = ["a, b", "b, c", "a", "c, a", "a, b, c", "b", "c", "b, a", "c, b", "a, c"]


def list_report(types):
    report = new_report("t", "x.csv")
    report["column_types"] = dict(types)
    return report


def test_list_column_split_into_its_own_table():
    frame = pd.DataFrame({"title": [f"t{i}" for i in range(10)], "tags": TAGS})
    report = list_report({"title": "text", "tags": "text"})

    frame, extra = split_list_columns(frame, report)

    assert frame["row_id"].tolist() == list(range(1, 11))
    child = extra[0]["cleaned"]
    assert extra[0]["table_name"] == "t_tags" and list(child.columns) == ["row_id", "tags_value"]
    assert len(child) == 18 and set(child["tags_value"]) == {"a", "b", "c"}


def test_existing_row_id_is_never_reused_and_keep_skips_split():
    frame = pd.DataFrame({"row_id": [1] * 10, "tags": TAGS})
    report = list_report({"row_id": "integer", "tags": "text"})
    split, extra = split_list_columns(frame.copy(), report)
    assert list(extra[0]["cleaned"].columns) == ["_row_id", "tags_value"]

    report = list_report({"row_id": "integer", "tags": "text"})
    kept, extra = split_list_columns(frame.copy(), report, {"tags": {"list_column": "keep"}})
    assert extra == [] and list(kept.columns) == ["row_id", "tags"]
    assert any(flag["issue"] == "list_column" for flag in report["flags"])


def test_free_text_is_not_a_list_column():
    notes = [f"Called on day {i}, left a message, will retry" for i in range(12)]
    frame = pd.DataFrame({"notes": notes})

    assert split_list_columns(frame, list_report({"notes": "text"}))[1] == []


# ------------------------------------------------------------
# Choices applied by the cleaner
# ------------------------------------------------------------

def messy_table():
    rows = [["North", "10", "New Zealand", "1.007.598"],
            ["South", "", "New Zealand", "2.000"],
            ["North", "30", "New Zeland", "abc"],
            ["West", "20", "India", "4"],
            ["South", "", "India", "5"],
            ["North", "40", "India", "6"],
            ["West", "50", "India", "7"],
            ["North", "60", "India", "8"],
            ["South", "70", "India", "9"],
            ["West", "80", "India", "10"]]
    return grid(["region", "units", "country", "votes"], rows)


def test_without_choices_only_safe_fixes_run():
    frame, _, report = clean_table(messy_table(), "t", "t.csv")
    issues = {(flag["column"], flag["issue"]) for flag in report["flags"]}

    assert frame["units"].isna().sum() == 2
    assert ("units", "missing_values") in issues
    assert ("country", "similar_categories") in issues
    assert ("votes", "partly_numeric") in issues
    assert report["column_types"]["votes"] == "text"


def test_choices_fill_merge_and_convert():
    choices = {"units": {"missing_values": "fill_median"},
               "country": {"similar_categories": {"New Zeland": "New Zealand"}},
               "votes": {"partly_numeric": "convert"}}

    frame, _, report = clean_table(messy_table(), "t", "t.csv", choices)

    assert frame["units"].isna().sum() == 0 and frame["units"].iloc[1] == 45
    assert set(frame["country"]) == {"New Zealand", "India"}
    assert report["column_types"]["votes"] == "integer"
    assert frame["votes"].iloc[0] == 1007598 and pd.isna(frame["votes"].iloc[2])
    assert sum("your choice" in action["detail"] for action in report["actions"]) == 3


def test_drop_rows_choice_and_all_rows_removed():
    frame, _, _ = clean_table(messy_table(), "t", "t.csv",
                              {"units": {"missing_values": "drop_rows"}})
    assert len(frame) == 8


# ------------------------------------------------------------
# Review items
# ------------------------------------------------------------

def review_items():
    _, _, report = clean_table(messy_table(), "t", "t.csv")
    report["rows_before"] = 10
    return build_review([report])


def test_review_items_have_options_and_suggestions():
    items = {(item["column"], item["issue"]): item for item in review_items()}

    units = items[("units", "missing_values")]
    assert units["suggested"] == "leave"            # 20% missing: too many to fill
    assert "fill_median" in [option["value"] for option in units["options"]]

    assert items[("votes", "partly_numeric")]["suggested"] == "convert"
    assert items[("country", "similar_categories")]["groups"] == {"New Zeland": "New Zealand"}


def test_suggested_and_validated_choices():
    items = review_items()
    choices = suggested_choices(items)

    assert choices["t"]["country"]["similar_categories"] == {"New Zeland": "New Zealand"}
    assert validate_choices(choices, items) == choices

    with pytest.raises(ValueError):
        validate_choices({"t": {"units": {"missing_values": "fill_unknown"}}}, items)
    with pytest.raises(ValueError):
        validate_choices({"t": {"nope": {"missing_values": "leave"}}}, items)


# ------------------------------------------------------------
# Long summaries and "top 5 rows"
# ------------------------------------------------------------

def test_summary_that_lists_every_row_is_sent_back():
    frame = pd.DataFrame({"year": range(1950, 1970), "genre": ["Drama"] * 20})
    long = " ".join(f"In {year} it was Drama." for year in range(1950, 1970))

    assert "Too long" in summary_problem(long, frame)
    assert summary_problem("Drama led in every year from 1950 to 1969.", frame) is None


@pytest.mark.parametrize("message, expected", [
    ("top 5 rows", {"rows": 5, "from_end": False}),
    ("show me the last 3 records", {"rows": 3, "from_end": True}),
    ("show the data", {"rows": 5, "from_end": False}),
    ("give me 500 rows", {"rows": 50, "from_end": False}),
    ("top 5 rows of that", None),
    ("top 5 products", None),
    ("show me", None),
])
def test_row_preview_request(message, expected):
    assert row_preview_request(message) == expected


def test_naming_a_column_is_still_a_data_question():
    assert classify_message("first 5 rows by region", ["region"])["intent"] == "data"


def test_resolved_flags_give_no_caveat():
    flag = {"column": "country", "issue": "similar_categories",
            "detail": "Possibly the same category: [('US', 'USA')]; not merged."}
    profile = {"tables": {"t": {"table_name": "t", "flags": [flag]}}}
    sql = "SELECT country, COUNT(*) FROM t GROUP BY country"

    assert build_caveats(sql, profile)
    flag["resolved"] = True
    assert build_caveats(sql, profile) == []


def test_ids_and_codes_are_not_suggested_as_amounts(tmp_path):
    import sqlite3
    from src.helpers.dataset_preview import suggested_questions

    db = tmp_path / "d.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE t (row_id INTEGER, pin INTEGER, price INTEGER, region TEXT)")
    connection.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [
        (i, [110001, 400001, 560001][i % 3], 100 * i + 7, ["N", "S", "W"][i % 3])
        for i in range(1, 11)])
    connection.commit()
    connection.close()

    def column(kind, unique, low, high, **extra):
        return {"kind": kind, "unique": unique, "min": low, "max": high,
                "missing_pct": 0, **extra}

    table = {"rows": 10, "columns": {
        "row_id": column("integer", 10, 1, 10),
        "pin": column("integer", 3, 110001, 560001),
        "price": column("integer", 10, 107, 1007),
        "region": column("text", 3, None, None, is_category=True),
    }}
    questions = suggested_questions(db, "t", table, "orders")

    assert "Average price by region, as a bar chart" in questions
    assert "Share of orders by region, as a pie chart" in questions
    assert not any("row id" in q or "pin" in q for q in questions)


def test_only_chartable_results_are_suggested():
    from src.helpers.dataset_preview import result_is_chartable

    assert result_is_chartable([("N", 1), ("S", 2), ("W", 3)], "pie")
    assert not result_is_chartable([("N", 1)], "bar")                     # one bar
    assert not result_is_chartable([(str(i), i) for i in range(30)], "bar")  # too many
    assert not result_is_chartable([(str(i), i) for i in range(9)], "pie")   # too many slices
    assert not result_is_chartable([("2020", 4), ("2021", 4), ("2022", 4)], "line")  # flat
    assert not result_is_chartable([("N", None), ("S", None)], "bar")      # all empty


def test_row_noun_falls_back_to_rows(monkeypatch):
    import src.helpers.dataset_preview as preview

    monkeypatch.setattr(preview, "chat_json", lambda *a, **k: ({"noun": "movies"}, {}))
    assert preview.name_rows({}, "t", ["a"], [[1]]) == "movies"

    for odd in ("Movies!", "the rows of this table are movies", ""):
        monkeypatch.setattr(preview, "chat_json", lambda *a, odd=odd, **k: ({"noun": odd}, {}))
        assert preview.name_rows({}, "t", ["a"], [[1]]) == "rows"

    def down(*args, **kwargs):
        raise ValueError("no JSON")

    monkeypatch.setattr(preview, "chat_json", down)
    assert preview.name_rows({}, "t", ["a"], [[1]]) == "rows"


def test_summary_naming_something_not_in_the_result_is_sent_back():
    frame = pd.DataFrame({"country": ["USA", "UK", "New Zeland"], "movies": [64, 12, 3]})

    bad = "USA leads with 64. Germany and Canada (misspelled as New Zeland) have 3."
    assert "Canada" in summary_problem(bad, frame, "movies by country")
    good = "The USA has the most movies with 64, followed by the UK with 12."
    assert summary_problem(good, frame, "movies by country") is None
    # Words from the question are fine; so are month names.
    assert summary_problem("In March the USA had 64 Movies.", frame, "movies") is None


def test_model_picks_spelling_only_from_the_group(monkeypatch):
    import src.helpers.cleaning_review as review

    rule = {"New Zealand": "New Zeland", "New Zesland": "New Zeland", "US": "USA"}

    monkeypatch.setattr(review, "chat_json",
                        lambda *a, **k: ({"picks": ["New Zealand", "USA"]}, {}))
    assert review.pick_spellings({}, "country", rule) == {
        "New Zeland": "New Zealand", "New Zesland": "New Zealand", "US": "USA"}

    # A pick outside the group, or no model: the rule's pick stays.
    monkeypatch.setattr(review, "chat_json", lambda *a, **k: ({"picks": ["Canada"]}, {}))
    assert review.pick_spellings({}, "country", rule) == rule

    def down(*args, **kwargs):
        raise ValueError("no JSON")

    monkeypatch.setattr(review, "chat_json", down)
    assert review.pick_spellings({}, "country", rule) == rule


@pytest.mark.parametrize("raw, number", [
    ("178c", 178), ("$ 4o8,035,783", 408035783), ("9,.0", 9.0), ("8:8", 8.8),
    ("++8.7", 8.7), ("8.7.", 8.7), ("8,7e-0", 8.7), ("8..8", 8.8),
])
def test_typos_in_numbers_are_repaired(raw, number):
    from src.helpers.data_cleaner import repair_number

    assert repair_number(raw)[0] == number


def test_words_are_not_repaired_and_ids_keep_their_zeros():
    from src.helpers.data_cleaner import repair_number, try_number

    assert repair_number("Inf") is None and repair_number("Not Applicable") is None
    assert try_number(pd.Series(["00123", "00456", "00789"] * 4))["kind"] == "leading_zero"


def test_number_repairs_are_listed_and_can_be_undone():
    rows = [[f"t{i}", str(7 + i / 10)] for i in range(18)] + [["a", "8..8"], ["b", "08.9"]]
    table = grid(["title", "score"], rows)

    frame, _, report = clean_table(table, "t", "t.csv")
    flag = next(f for f in report["flags"] if f["issue"] == "numbers_guessed")
    assert report["column_types"]["score"] == "float" and flag["guesses"] == [["8..8", 8.8]]
    assert frame["score"].iloc[18] == 8.8 and frame["score"].iloc[19] == 8.9

    frame, _, _ = clean_table(table, "t", "t.csv", {"score": {"numbers_guessed": "empty"}})
    assert pd.isna(frame["score"].iloc[18]) and frame["score"].iloc[19] == 8.9


def test_a_decimal_comma_column_is_not_guessed():
    from src.helpers.data_cleaner import try_number

    assert try_number(pd.Series(["12,5", "3,25", "7,0", "1,5"] * 3))["kind"] == "decimal_comma"


def test_mixed_encoding_is_repaired():
    from src.helpers.data_cleaner import repair_mojibake

    assert repair_mojibake("Per qualche dollaro in pi\u00c3\u00b9") == "Per qualche dollaro in pi\u00f9"
    assert repair_mojibake("L\u00c3\u00a9on") == "L\u00e9on"
    for clean in ("L\u00e9on", "K\u00e1tia Lund", "Plain text", None, 5):
        assert repair_mojibake(clean) == clean


def test_same_meaning_needs_two_yes_answers_and_small_groups(monkeypatch):
    import src.helpers.cleaning_review as review

    def model(answers):
        replies = iter(answers)
        return lambda *args, **kwargs: (next(replies), {})

    values = ["Approved", "G", "Not Rated", "PG", "R", "Unrated"]
    monkeypatch.setattr(review, "chat_json", model([
        {"pairs": [{"a": "Not Rated", "b": "Unrated", "keep": "Unrated"}]}, {"same": True}]))
    assert review.column_synonyms({}, "rating", values) == {"Not Rated": "Unrated"}

    # The second opinion says no: nothing.
    monkeypatch.setattr(review, "chat_json", model([
        {"pairs": [{"a": "G", "b": "PG", "keep": "PG"}]}, {"same": False}]))
    assert review.column_synonyms({}, "rating", values) == {}

    # Values that do not exist, or one big "group": nothing.
    monkeypatch.setattr(review, "chat_json", model([
        {"pairs": [{"a": "Unrated", "b": "Unknown", "keep": "Unknown"}]}]))
    assert review.column_synonyms({}, "rating", values) == {}
    big = [{"a": value, "b": "R", "keep": "R"} for value in values if value != "R"]
    monkeypatch.setattr(review, "chat_json", model([{"pairs": big}] + [{"same": True}] * 9))
    assert review.column_synonyms({}, "rating", values) == {}


def test_confirmed_same_meaning_is_merged_by_the_cleaner():
    rows = [[f"t{i}", rating] for i, rating in
            enumerate(["R", "PG", "Not Rated", "Unrated", "R", "PG", "Unrated", "R", "PG", "R"])]
    table = grid(["title", "rating"], rows)

    frame, _, report = clean_table(table, "t", "t.csv",
                                   {"rating": {"same_meaning": {"Not Rated": "Unrated"}}})

    assert set(frame["rating"]) == {"R", "PG", "Unrated"}
    assert any(a["action"] == "merged_same_meaning" for a in report["actions"])


def test_same_meaning_keep_given_as_a_letter(monkeypatch):
    import src.helpers.cleaning_review as review

    replies = iter([{"pairs": [{"a": "Not Rated", "b": "Unrated", "keep": "b"}]}, {"same": True}])
    monkeypatch.setattr(review, "chat_json", lambda *a, **k: (next(replies), {}))

    assert review.column_synonyms({}, "rating", ["G", "Not Rated", "PG", "R", "Unrated"]) == \
        {"Not Rated": "Unrated"}



@pytest.mark.parametrize("text, number", [
    ("two hundred", 200), ("fifty-five", 55), ("one thousand two hundred and five", 1205),
    ("three lakh", 300000), ("1.5k", 1500), ("2 crore", 20000000), ("twelve", 12),
])
def test_written_numbers_are_read(text, number):
    from src.helpers.data_cleaner import words_to_number

    assert words_to_number(text) == number


def test_written_numbers_in_a_column_are_listed():
    from src.helpers.data_cleaner import try_number, words_to_number

    for not_a_number in ("confidential", "N/A", "two apples", "Product A"):
        assert words_to_number(not_a_number) is None

    result = try_number(pd.Series(["300", "two hundred", "100", "50", "two hundred"]))
    assert result["kind"] == "integer" and result["values"].tolist() == [300, 200, 100, 50, 200]
    assert ("two hundred", 200) in result["guessed"]



def test_prefixed_words_are_not_look_alikes():
    from src.helpers.data_cleaner import prefixed_pair

    for different in (("Male", "Female"), ("Active", "Inactive"), ("Paid", "Unpaid"),
                      ("profit", "Non-profit"), ("Legal", "Illegal"), ("Regular", "Irregular")):
        assert prefixed_pair(*different)
    for typo in (("New Zealand", "New Zeland"), ("US", "USA"), ("Italy", "Italy1")):
        assert not prefixed_pair(*typo)

    rows = [[f"p{i}", gender] for i, gender in
            enumerate(["Male", "Female", "Other", "Male", "Female", "Male", "Female", "Male",
                       "Female", "Male", "Femal"])]
    _, _, report = clean_table(grid(["person", "gender"], rows), "t", "t.csv")
    flags = [f for f in report["flags"] if f["issue"] == "similar_categories"]
    assert flags and all(("Female", "Male") != tuple(pair) and ("Male", "Female") != tuple(pair)
                         for pair in flags[0]["pairs"])



def test_a_choice_equal_to_the_suggestion_still_counts_as_decided():
    from src.helpers.data_cleaner import mark_resolved

    report = {"flags": [{"column": "d", "issue": "dates_guessed"},
                        {"column": "x", "issue": "outliers"},
                        {"column": "y", "issue": "missing_values"}]}
    mark_resolved(report, {"d": {"dates_guessed": "keep_dates"},
                           "x": {"outliers": "set_null"}})

    flags = {flag["column"]: flag for flag in report["flags"]}
    assert flags["d"]["decided"] and not flags["d"]["resolved"]
    assert flags["x"]["decided"] and flags["x"]["resolved"]
    assert not flags["y"]["decided"] and not flags["y"]["resolved"]
