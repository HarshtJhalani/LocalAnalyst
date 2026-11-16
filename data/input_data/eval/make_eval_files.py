"""
Generate the messy files for the cleaning evaluation (notebook 08) and
their answer key. Problems differ from the sample files used to build
notebook 01, so the evaluation is not graded on data it was tuned on.

Run: python make_eval_files.py <output folder>
"""

import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from openpyxl import Workbook

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

# ------------------------------------------------------------
# eval_inventory.csv: ';' delimiter, cp1252 encoding, two title
# lines, "Rs" prices, % margins, YYYY/MM/DD dates, Y/N flags
# ------------------------------------------------------------
categories = ["Beverages", "beverages ", "BEVERAGES", "Snacks", "Dairy"]
names = ["Café Latte", "Masala Chai", "Crème Biscuit", "Paneer Block",
         "Lemon Soda", "Salted Chips", "Greek Yogurt", "Cold Coffee"]

rows = []
for i in range(24):
    rows.append([
        f"A-{i + 1:03d}",
        random.choice(names),
        random.choice(categories),
        str(random.randint(5, 400)),
        f"Rs {random.randint(20, 2500):,}.{random.choice(['00', '50', '75'])}",
        f"{random.choice([5, 7.5, 10, 12.5, 15])}%",
        (date(2024, 1, 1) + timedelta(days=random.randint(0, 200))).strftime("%Y/%m/%d"),
        random.choice(["Y", "N", "yes", "no"]),
        "",
    ])

rows[4][3] = "N/A"
rows[9][3] = "--"
rows[15][3] = ""
rows[20][3] = "99999"                        # outlier
rows[2][8] = "FreshCo"
rows[11][8] = "FreshCo"

header = ["Item Code", "Item Name", "Category", "Stock Qty", "Unit Cost (Rs)",
          "Margin %", "Last Restock", "Discontinued?", "Supplier"]

with open(OUT / "eval_inventory.csv", "w", newline="", encoding="cp1252") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["Inventory export"])
    w.writerow(["Warehouse 3"])
    w.writerow(header)
    for n, row in enumerate(rows):
        w.writerow(row)
        if n == 7:
            w.writerow(row)                  # duplicate
        if n == 13:
            w.writerow([""] * len(header))   # empty row

# ------------------------------------------------------------
# eval_tickets.xlsx: US month-first dates as text, IDs with
# leading zeros, TRUE/FALSE, one fully empty column
# ------------------------------------------------------------
wb = Workbook()
ws = wb.active
ws.title = "Tickets"
ws.append(["Ticket ID", "Opened", "Priority", "Resolved", "Hours Spent", "Notes"])
for i in range(18):
    opened = date(2024, 9, 1) + timedelta(days=random.randint(0, 100))
    ws.append([
        f"{i + 7:04d}",                                  # 0007, 0008 ...
        opened.strftime("%m/%d/%Y"),                     # month-first text
        random.choice(["High", "high", "Medium", "Low"]),
        random.choice(["TRUE", "FALSE"]),
        round(random.uniform(0.5, 12), 1),
        None,
    ])
ws.cell(row=6, column=5, value="unknown")              # one non-number
wb.save(OUT / "eval_tickets.xlsx")

# ------------------------------------------------------------
# Answer key
# ------------------------------------------------------------
truth = {
    "eval_inventory.csv": {
        "tables": {"eval_inventory": {
            "header_row": 2,
            "rows_after": 24,
            "columns": ["item_code", "item_name", "category", "stock_qty",
                        "unit_cost", "margin_pct", "last_restock",
                        "discontinued", "supplier"],
            "types": {"item_code": "text", "item_name": "text", "category": "text",
                      "stock_qty": "integer", "unit_cost": "float",
                      "margin_pct": "float", "last_restock": "date",
                      "discontinued": "boolean", "supplier": "text"},
            "categories": {"category": ["Beverages", "Dairy", "Snacks"]},
            "missing": {"stock_qty": 3},
            "flags": [["stock_qty", "outliers"], ["supplier", "missing_values"]],
            "text_contains": {"item_name": "Café Latte"},
        }},
    },
    "eval_tickets.xlsx": {
        "tables": {"eval_tickets": {
            "header_row": 0,
            "rows_after": 18,
            "columns": ["ticket_id", "opened", "priority", "resolved", "hours_spent"],
            "types": {"ticket_id": "text", "opened": "date", "priority": "text",
                      "resolved": "boolean", "hours_spent": "float"},
            "categories": {"priority": ["High", "Low", "Medium"]},
            "missing": {"hours_spent": 1},
            "flags": [["ticket_id", "identifier_kept_as_text"]],
            "text_contains": {},
        }},
    },
}
(OUT / "eval_cleaning_truth.json").write_text(json.dumps(truth, indent=2),
                                              encoding="utf-8")
print("written:", sorted(p.name for p in OUT.iterdir()))
