"""Generate deliberately messy sample files for notebook 01."""

import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
random.seed(7)

# ------------------------------------------------------------
# messy_sales.csv
# ------------------------------------------------------------
products = {"Laptop": 55000, "Monitor": 12000, "Keyboard": 1500,
            "Mouse": 800, "Headphones": 3500}
cities = ["Delhi", "delhi", " Delhi ", "DELHI", "Mumbai", "mumbai",
          "Mumbay", "Bengaluru", "Pune", "New Delhi"]
regions = {"delhi": "North", "new delhi": "North", "mumbai": "West",
           "mumbay": "West", "bengaluru": "South", "pune": "West"}
names = ["Asha Verma", "Rahul Singh", "Priya Nair", "Vikram Rao",
         "Neha Gupta", "Arjun Mehta", "Kavya Iyer", "Rohan Das",
         "Sneha Kulkarni", "Amit Shah"]
reps = ["Kiran", "Meera", "Sanjay"]

rows = []
start = date(2024, 1, 2)
for i in range(32):
    product = random.choice(list(products))
    city = random.choice(cities)
    units = random.randint(1, 10)
    discount = random.choice([0, 5, 10])
    price = products[product]
    total = units * price * (1 - discount / 100)
    rows.append({
        "id": f"00{101 + i}",
        "date": (start + timedelta(days=random.randint(0, 88))).strftime("%d/%m/%Y"),
        "customer": random.choice(names),
        "city": city,
        "region": regions[city.strip().casefold()],
        "product": product,
        "units": str(units),
        "price": f"${price:,.2f}",
        "discount": f"{discount}%",
        "total": f"${total:,.2f}",
        "paid": random.choice(["Yes", "No", "Y", "N", "yes"]),
        "rep": random.choice(reps),
        "backup": random.choice(reps),
        "notes": "",
    })

# Planted problems
rows[3]["units"] = "N/A"
rows[8]["units"] = "-"
rows[12]["units"] = "twelve"
rows[5]["customer"] = "  Asha   Verma "
rows[15]["price"] = ""
rows[20]["total"] = "$9,999,999.00"          # outlier
rows[2]["notes"] = "Gift wrap"
rows[17]["notes"] = "Late delivery"
rows[25]["notes"] = "null"

header = ["Order ID", " Order Date ", "Customer Name", "City", "Region",
          "Product", "Units Sold", "Unit Price ($)", "Discount %",
          "Total Sales ($)", "Paid?", "Sales Rep", "Sales Rep ", "Notes", ""]

with open(OUT / "messy_sales.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Quarterly Sales Report - Q1 2024"])
    w.writerow(["Generated: 2024-04-01"])
    w.writerow([])
    w.writerow(header)
    for n, r in enumerate(rows):
        line = [r["id"], r["date"], r["customer"], r["city"], r["region"],
                r["product"], r["units"], r["price"], r["discount"],
                r["total"], r["paid"], r["rep"], r["backup"], r["notes"], ""]
        w.writerow(line)
        if n in (6, 19):                       # exact duplicate rows
            w.writerow(line)
        if n == 10:                            # empty row in the middle
            w.writerow([""] * len(header))

# ------------------------------------------------------------
# messy_employees.xlsx (two sheets)
# ------------------------------------------------------------
wb = Workbook()
ws = wb.active
ws.title = "Employees"
ws.append(["Employee Master"])
ws.append([])
ws.append(["Emp ID", "Full Name", "Department", "Joining Date",
           "Salary (INR)", "Last Review", "Active"])
departments = ["Sales", "sales ", "HR", "hr", "Engineering", "Finance"]
for i in range(20):
    salary = random.randint(35, 120) * 1000
    ws.append([
        f"E{200 + i}",
        f"Employee {i + 1}",
        random.choice(departments),
        date(2018, 1, 1) + timedelta(days=random.randint(0, 2000)),
        "confidential" if i in (4, 11) else salary,
        f"{random.randint(1, 12):02d}/{random.randint(1, 12):02d}/2024",  # ambiguous
        random.choice(["TRUE", "FALSE"]),
    ])

ws2 = wb.create_sheet("Targets")
ws2.append(["Department", "Q1 Target", "Q2 Target"])
for dept, q1, q2 in [("Sales", 500000, 550000), ("HR", 0, 0),
                     ("Engineering", 0, 0), ("Finance", 100000, 120000)]:
    ws2.append([dept, q1, q2])
wb.save(OUT / "messy_employees.xlsx")

# ------------------------------------------------------------
# customers.json (nested records)
# ------------------------------------------------------------
customers = []
for i, name in enumerate(names[:8]):
    customers.append({
        "customer_id": 1 + i,
        "name": name,
        "signup": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "address": {"city": random.choice(["Delhi", "Mumbai", "Pune"]),
                    "pin": str(random.choice([110001, 400001, 411001]))},
        "tags": random.sample(["vip", "new", "bulk", "online"], 2),
    })
(OUT / "customers.json").write_text(json.dumps({"customers": customers}, indent=2),
                                    encoding="utf-8")

# ------------------------------------------------------------
# products.parquet (already typed)
# ------------------------------------------------------------
pd.DataFrame({
    "product": list(products),
    "category": ["Computers", "Computers", "Accessories", "Accessories", "Audio"],
    "list_price": list(products.values()),
}).to_parquet(OUT / "products.parquet", index=False)

print("written:", sorted(p.name for p in OUT.iterdir()))
