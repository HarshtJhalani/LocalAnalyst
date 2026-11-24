
# ============================================================
# FILE: tests/test_chart_requests.py
# PURPOSE: Chart types named in the question (pie, donut, column,
#          area, stacked, ...) are drawn when the result fits them,
#          and fall back with a note when it does not
# ============================================================

import pandas as pd
import pytest

from src.helpers.chart_builder import build_figure, choose_chart, requested_chart

REPS = pd.DataFrame({"sales_rep": ["Kiran", "Meera", "Sanjay"],
                     "total_units_sold": [49, 54, 61]})
MONTHS = pd.DataFrame({"month": ["2024-01", "2024-02", "2024-03"],
                       "total_sales": [10.0, 12.0, 9.0]})
TWO_GROUPS = pd.DataFrame({"region": ["North", "North", "West", "West"],
                           "product": ["Laptop", "Mouse", "Laptop", "Mouse"],
                           "total_sales": [100, 20, 80, 30]})


@pytest.mark.parametrize("question, expected", [
    ("pie chart for the above", "pie"),
    ("that is not a pie chart", "pie"),
    ("make it a doughnut", "donut"),
    ("stacked bar of sales by region and product", "stacked_bar"),
    ("area chart of monthly sales", "area"),
    ("column chart please", "column"),
    ("show it as a line chart", "line"),
    ("bar plot of units_sold by region", "bar"),
    ("histogram of prices", "histogram"),
    ("give the column names", None),
    ("sales by product line", None),
    ("total sales by region", None),
])
def test_requested_chart(question, expected):
    assert requested_chart(question) == expected


def test_pie_and_donut_drawn_when_asked():
    pie = choose_chart(REPS, "units sold by each sales rep", "pie chart for the above")
    donut = choose_chart(REPS, "units sold by each sales rep as a donut")

    assert pie["type"] == "pie" and pie["x"] == "sales_rep"
    assert donut["type"] == "donut"
    assert choose_chart(REPS, "units sold by each sales rep")["type"] == "bar"   # default

    for spec in (pie, donut):
        for mode in ("light", "dark"):
            figure = build_figure(REPS, spec, mode)
            assert figure.data[0].type == "pie"
            assert list(figure.data[0].labels) == ["Sanjay", "Meera", "Kiran"]   # largest first


def test_pie_falls_back_when_it_would_mislead():
    negative = REPS.assign(total_units_sold=[49, -5, 61])
    spec = choose_chart(negative, "pie chart of value by name")

    assert spec["type"] == "bar" and "pie chart" in spec["note"]


def test_pie_with_many_parts_groups_the_rest_as_other():
    many = pd.DataFrame({"name": list("ABCDEFGH"), "value": range(8, 0, -1)})
    spec = choose_chart(many, "pie chart of value by name")

    assert spec["type"] == "pie" and spec["other_after"] == 5 and "Other" in spec["note"]
    pie = build_figure(many, spec, "light").data[0]
    assert list(pie.labels) == ["A", "B", "C", "D", "E", "Other"]
    assert list(pie.values)[-1] == 3 + 2 + 1 and sum(pie.values) == sum(range(1, 9))


def test_other_requested_types():
    assert choose_chart(REPS, "column chart of units by rep")["type"] == "column"
    assert choose_chart(MONTHS, "area chart of monthly sales")["type"] == "area"
    assert choose_chart(MONTHS, "monthly sales as a bar chart")["type"] == "bar"
    assert choose_chart(TWO_GROUPS, "stacked bar of sales by region and product")["type"] \
        == "stacked_bar"

    no_dates = choose_chart(REPS, "line chart of units by rep")
    assert no_dates["type"] == "bar" and "date" in no_dates["note"]

    for spec, frame in ((choose_chart(REPS, "column chart"), REPS),
                        (choose_chart(MONTHS, "area chart"), MONTHS),
                        (choose_chart(TWO_GROUPS, "stacked chart"), TWO_GROUPS)):
        assert build_figure(frame, spec, "dark").data



def test_bar_asked_with_whole_numbers_as_labels():
    frame = pd.DataFrame({"age": [None, 25.0, 30, 35, 40, 50],
                          "avg_salary": [59000, 60479.45, 60436.24, 60150, 60280, 60530.3]})
    spec = choose_chart(frame, "average salary vs age bar plot")

    assert spec["type"] == "bar" and spec["y"] == "age" and spec["x"] == "avg_salary"
    labels = list(build_figure(frame, spec, "light").data[0].y)
    assert "25" in labels and "(blank)" in labels and "25.0" not in labels

    # Without a bar request, two number columns stay a scatter.
    assert choose_chart(frame, "average salary vs age")["type"] == "scatter"
