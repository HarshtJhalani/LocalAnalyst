
# ============================================================
# FILE: tests/test_message_intent.py
# PURPOSE: Messages that are not data questions are sorted out before
#          any SQL: vague, feedback, chart change
# ============================================================

import pytest

from src.helpers.message_intent import classify_message

COLUMNS = ["order_id", "order_date", "region", "product", "units_sold",
           "total_sales", "sales_rep"]


@pytest.mark.parametrize("message, intent, chart", [
    ("what", "vague", None),
    ("huh?", "vague", None),
    ("?", "vague", None),
    ("ok thanks", "vague", None),
    ("that's wrong", "feedback", None),
    ("not what I asked", "feedback", None),
    ("that is not a pie chart", "chart_change", "pie"),
    ("pie chart for the above", "chart_change", "pie"),
    ("make it a donut", "chart_change", "donut"),
    ("show it as bars", "chart_change", "bar"),
    ("line chart", "chart_change", "line"),
    # Data questions, even short or with a chart type in them.
    ("Total sales by region", "data", None),
    ("units sold by sales rep as a pie chart", "data", "pie"),
    ("Laptop only", "data", None),
    ("Only for Laptop.", "data", None),
    ("How many rows are there?", "data", None),
    ("what columns are there?", "data", None),
    # Requests to change the data are refused before any SQL.
    ("delete the North rows", "write_request", None),
    ("Delete rows where region is North", "write_request", None),
    ("please update units_sold to 5", "write_request", None),
    ("can you drop the table", "write_request", None),
    # ... but analytical wording that starts the same way is not.
    ("change in sales by month", "data", None),
    ("add up total sales", "data", None),
    ("which orders were removed", "data", None),
    ("how many rows were deleted", "data", None),
    ("change it to bars", "chart_change", "bar"),
    ("fix the chart", "feedback", None),
])
def test_classify_message(message, intent, chart):
    result = classify_message(message, COLUMNS)

    assert result["intent"] == intent
    assert result["chart"] == chart
    assert (result["reply"] is not None) == (intent in {"vague", "feedback", "write_request"})


def test_intent_words_in_column_names_do_not_hijack_messages():
    columns = ["region", "last", "change", "chart_type", "total_sales"]

    assert classify_message("make it a pie chart instead", columns)["intent"] == "chart_change"
    assert classify_message("change it to bars", columns)["intent"] == "chart_change"
    assert classify_message("that's wrong", columns)["intent"] == "feedback"
    # The column itself can still be asked about.
    assert classify_message("chart_type counts", columns)["intent"] == "data"
    assert classify_message("total sales by region", columns)["intent"] == "data"



@pytest.mark.parametrize("message, expected", [
    ("fill age with the median", ("age", "fill_median", False)),
    ("replace missing salary with the mean", ("salary", "fill_mean", False)),
    ("Apply: fill missing units_sold with the median", ("units_sold", "fill_median", True)),
    ("drop rows with missing email", ("email", "drop_rows", False)),
    ("delete rows where email is missing", ("email", "drop_rows", False)),
    ("Delete rows where region is North", None),
    ("delete the North rows", None),
    ("What is the median age?", None),
])
def test_cleaning_requests(message, expected):
    from src.helpers.message_intent import cleaning_request

    found = cleaning_request(message)
    assert (found and (found["target"], found["option"], found["confirmed"])) == expected \
        if expected else found is None



@pytest.mark.parametrize("message, target, option", [
    ("fill median age in rows where age is empty", "age", "fill_median"),
    ("fill median age in rows where age is 0", "age", "fill_median"),
    ("impute missing ages using the mean", "ages", "fill_mean"),
    ("use the median for missing salary", "salary", "fill_median"),
    ("fill in the blanks in department with the most common value", "department",
     "fill_most_common"),
    ("fill missing units sold with 0", "units sold", "fill_zero"),
])
def test_fill_requests_in_any_word_order(message, target, option):
    from src.helpers.message_intent import cleaning_request

    found = cleaning_request(message)
    assert found["target"] == target and found["option"] == option


@pytest.mark.parametrize("message", [
    "What is the median age?", "Show the median salary by department",
    "use department as the grouping", "Set every order's paid flag to true",
])
def test_not_fill_requests(message):
    from src.helpers.message_intent import cleaning_request

    assert cleaning_request(message) is None
