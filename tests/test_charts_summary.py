
# ============================================================
# FILE: tests/test_charts_summary.py
# PURPOSE: chart_builder and answer_summary (checks from
#          06_charts_summary.ipynb); the summary model call is replaced
# ============================================================

import pandas as pd
import pytest

import src.helpers.answer_summary as answer_summary
from config import BAR_SLOT_PX, MAX_BARS
from src.helpers.answer_summary import build_caveats, summarize, unverified_numbers
from src.helpers.chart_builder import THEMES, build_figure, choose_chart, series_colors
from src.helpers.pipeline import load_dataset


@pytest.mark.parametrize("frame, chart", [
    (pd.DataFrame(), "none"),
    (pd.DataFrame({"orders": [32]}), "stat"),
    (pd.DataFrame({"month": ["2024-01", "2024-02"], "s": [1, 2]}), "line"),
    (pd.DataFrame({"month": ["2024-01", "2024-02"], "a": [1, 2], "b": [1000, 2000]}), "multiples"),
    (pd.DataFrame({"city": ["A", "B"], "s": [1, 2]}), "bar"),
    (pd.DataFrame({"r": ["N", "S"], "p": ["y", "n"], "s": [1, 2]}), "grouped_bar"),
    (pd.DataFrame({"a": range(6), "b": range(6)}), "scatter"),
    (pd.DataFrame({"a": range(20)}), "histogram"),
    (pd.DataFrame({"name": ["x", "y"]}), "table"),
    (pd.DataFrame({"year": [2023, 2024], "n": [1, 2]}), "line"),
])
def test_chart_choice(frame, chart):
    assert choose_chart(frame)["type"] == chart


def test_bars_largest_first_and_capped():
    many = pd.DataFrame({"c": [f"c{i}" for i in range(40)], "n": range(40)})
    figure = build_figure(many, choose_chart(many))

    assert len(figure.data[0].y) == MAX_BARS and figure.data[0].x[0] == 39
    assert figure.layout.height == 110 + BAR_SLOT_PX * MAX_BARS


def test_no_second_y_axis_and_themes():
    frame = pd.DataFrame({"month": ["2024-01", "2024-02"], "a": [1, 2], "b": [1000, 2000]})

    for mode in ("light", "dark"):
        layout = build_figure(frame, choose_chart(frame), mode).to_dict()["layout"]
        assert layout["paper_bgcolor"] == THEMES[mode]["surface"]
        assert not any(axis.get("overlaying") for key, axis in layout.items()
                       if key.startswith("yaxis") and isinstance(axis, dict))


def test_line_gaps_and_stable_colours():
    gappy = pd.DataFrame({"month": ["2024-01", "2024-03", "2024-01", "2024-02"],
                          "region": ["West", "West", "North", "North"], "n": [5, 3, 1, 2]})
    west = next(t for t in build_figure(gappy, choose_chart(gappy)).data if t.name == "West")

    assert list(west.x) == ["2024-01", "2024-02", "2024-03"] and pd.isna(west.y[1])
    assert series_colors(["Pune", "Delhi"], THEMES["light"])["Delhi"] == \
        series_colors(["Delhi"], THEMES["light"])["Delhi"]


def test_stat_tile_names_its_row():
    tile = build_figure(pd.DataFrame({"product": ["Monitor"], "units": [52]}),
                        {"type": "stat", "y": "units", "title": "t"})

    assert tile.data[0].title.text == "Monitor · Units"


def test_number_check():
    result = pd.DataFrame({"city": ["Delhi"], "total": [805358.857], "share": [0.25],
                           "month": ["2024-01"]})

    assert unverified_numbers("Delhi sold 805,358.86 (25%) in 2024.", result, "q") == []
    assert unverified_numbers("Delhi sold 805,359 units.", result, "q") == []
    assert unverified_numbers("Delhi sold 900,000.", result, "q") == [900000.0]
    assert unverified_numbers("The top 3 cities", result, "top 3 cities") == []


def test_summary_falls_back_to_template(monkeypatch):
    """A model that keeps inventing numbers ends with the fixed template."""

    monkeypatch.setattr(answer_summary, "chat_json",
                        lambda settings, messages, schema: ({"summary": "Sales were 123,456."}, {}))
    frame = pd.DataFrame({"city": ["Delhi", "Pune"], "sales": [10, 20]})
    result = summarize({}, "Sales by city?", "SELECT ...", frame, choose_chart(frame))

    assert result["source"] == "template" and result["attempts"] == 2
    assert result["summary"] == "The result has 2 rows. The highest sales is Pune with 20."


def test_caveats(sales):
    profile = load_dataset(sales)["profile"]

    assert any("9999999" in text for text in
               build_caveats("SELECT SUM(total_sales) FROM messy_sales", profile))
    assert any("Mumbay" in text for text in
               build_caveats("SELECT city FROM messy_sales", profile))
    assert build_caveats("SELECT COUNT(*) FROM messy_sales", profile) == []
