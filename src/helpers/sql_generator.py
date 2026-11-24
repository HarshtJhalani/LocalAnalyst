
# ============================================================
# FILE: src/helpers/sql_generator.py
# PURPOSE: Turn a question into one SQLite query with the model
#          (04_text_to_sql.ipynb; context argument from 07_memory.ipynb)
#
# Prompt: fixed rules (system), few-shot pairs on a small made-up
# `orders` table, then the dataset's schema text, optional conversation
# context and the question. Output JSON: plan (written first so the
# model names its columns and filters), can_answer, sql.
#
#   build_messages()                schema + question (+ context) -> messages
#   estimate_tokens()               messages -> rough prompt size
#   generate_sql_from_messages()    messages -> {"plan", "can_answer", "sql", ...}
# ============================================================

import json
import re
import logging

from config import PROMPT_CHARS_PER_TOKEN
from src.helpers.model_client import chat_json


logger = logging.getLogger(__name__)


# ============================================================
# 1. PROMPT
# ============================================================
# Tuned on data/input_data/dev_questions.json and measured on the
# held-out set in data/input_data/eval/ (see project_architecture.txt).

SYSTEM_PROMPT = """You are a data analyst who answers questions by writing one SQLite query.

Rules:
1. Write exactly one read-only query that starts with SELECT or WITH. If the question asks to change, delete or add data, set can_answer to false and say in plan that only reading data is possible.
2. Use only the tables and columns in the schema. Never invent columns.
3. Text comparisons are case-sensitive: copy category values exactly as listed in the schema.
4. Dates are TEXT 'YYYY-MM-DD'. Compare them as text, group them with strftime('%Y-%m', col) or strftime('%Y', col).
   "in 2023" means col >= '2023-01-01' AND col < '2024-01-01'; "after 2023" means col >= '2024-01-01'; "before 2023" means col < '2023-01-01'.
5. Boolean columns store 1 for true and 0 for false.
6. Give every computed column a short snake_case alias, e.g. SUM(total_sales) AS total_sales.
7. Group only by the columns the question asks about, and select them so the result is readable. When two columns have similar names (rep, rep_2), use the one without a number unless the question names the other.
8. For "top N", "most" or "highest" use ORDER BY ... DESC LIMIT N (LIMIT 1 for a single winner).
9. Use ROUND(value, 2) for averages and ratios.
10. If the question needs a value that no column holds (for example cost, profit, margin, rating, age), set can_answer to false, say which column is missing in plan, and leave sql empty. Never invent a formula from other columns to stand in for it.
11. "<number> by <category>" (also "plot/show ...") means SUM(<number>) per category, unless it asks for a number of rows ("how many orders") or an average. "<text column> by <category>" ("performance by department") counts each value per category: GROUP BY both, COUNT(*).
12. "missing", "blank", "empty", "not recorded" or "no <column>" mean <column> IS NULL (never = 0); count them with COUNT(*) ... WHERE <column> IS NULL.
13. "Which rows/orders/records ..." asks for the rows themselves: SELECT the key column(s) plus the asked column, with no GROUP BY.
14. Keep every grouping the question asks for: "by <X>" or "per <X>" needs GROUP BY <X>.
15. "How many <items> were sold / bought / ordered", "most sold" or "units sold" ask for a quantity: SUM(<quantity column>). COUNT(*) only counts rows (orders).
16. "Rank", "sort", "order" or "list <groups> by <measure>" returns every group, sorted, with no LIMIT.
17. "Which <group> has the highest / lowest <measure>" compares the groups: SUM(<measure>) per group (AVG when the question says average), never MIN or MAX of single rows.
18. "Average <measure> per order / per movie / per row" is AVG(<measure>) over all rows, with no GROUP BY. Add GROUP BY only for "by", "per <group>", "for each" or "in each".
19. "Has / had <column>": for a number column ("had a discount") <column> > 0; for a text column ("have notes") <column> IS NOT NULL.
20. A name, title or other text that is not listed exactly among a column's values (a surname, part of a title) is matched with LIKE '%text%'.
21. "Compare A and B" or "A vs B" returns only A and B: filter with <column> IN ('A', 'B').
22. "What / which is the longest / biggest / cheapest <thing>" asks for the thing: SELECT its name and the value, ORDER BY the value, LIMIT 1 (not MAX() alone).

Fill "plan" first: one or two sentences naming the table, columns, filters and grouping you will use."""


SQL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "can_answer": {"type": "boolean"},
        "sql": {"type": "string"},
    },
    "required": ["plan", "can_answer", "sql"],
}


EXAMPLE_SCHEMA = """TABLE orders (500 rows)
  order_id TEXT | unique per row | e.g. 'A001', 'A002', 'A003'
  order_date TEXT | date 'YYYY-MM-DD', 2023-01-01 to 2023-12-31
  store TEXT | values: 'Central', 'East', 'West'
  category TEXT | values: 'Books', 'Games', 'Toys'
  quantity INTEGER | range 1 to 20 | 4 missing
  amount REAL | currency amount | range 5.5 to 980.0
  returned INTEGER | boolean stored as 1 = true, 0 = false"""


FEW_SHOT_EXAMPLES = [
    ("How many games were sold?",
     {"plan": "Sold asks for a quantity: sum quantity of orders where category is 'Games' "
              "(COUNT would give the number of orders, not items).",
      "can_answer": True,
      "sql": "SELECT SUM(quantity) AS games_sold FROM orders WHERE category = 'Games'"}),

    ("How many orders came from the East store?",
     {"plan": "Count rows of orders where store is 'East'.",
      "can_answer": True,
      "sql": "SELECT COUNT(*) AS orders FROM orders WHERE store = 'East'"}),

    ("What is the total amount per category, highest first?",
     {"plan": "Sum amount from orders grouped by category, sorted descending.",
      "can_answer": True,
      "sql": "SELECT category, SUM(amount) AS total_amount FROM orders "
             "GROUP BY category ORDER BY total_amount DESC"}),

    ("Show the monthly number of returned orders.",
     {"plan": "Filter orders to returned = 1, group by month of order_date and count.",
      "can_answer": True,
      "sql": "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS returned_orders "
             "FROM orders WHERE returned = 1 GROUP BY month ORDER BY month"}),

    ("Which store has the highest median order amount?",
     {"plan": "Median of amount per store with median_of(json_group_array(amount)), keep the top one.",
      "can_answer": True,
      "sql": "SELECT store, median_of(json_group_array(amount)) AS median_amount "
             "FROM orders GROUP BY store ORDER BY median_amount DESC LIMIT 1"}),

    ("What was the average quantity of Toys orders in March 2023?",
     {"plan": "Average quantity of orders with category 'Toys' and order_date in 2023-03.",
      "can_answer": True,
      "sql": "SELECT ROUND(AVG(quantity), 2) AS average_quantity FROM orders "
             "WHERE category = 'Toys' AND order_date >= '2023-03-01' "
             "AND order_date < '2023-04-01'"}),

    ("How many orders were placed after 2022?",
     {"plan": "After 2022 means from 2023-01-01 onwards (2022 itself excluded), so count orders with order_date >= '2023-01-01'.",
      "can_answer": True,
      "sql": "SELECT COUNT(*) AS orders FROM orders WHERE order_date >= '2023-01-01'"}),

    ("Which customer spent the most?",
     {"plan": "The orders table has no customer column, so this cannot be answered.",
      "can_answer": False,
      "sql": ""}),

    ("Plot quantity by store.",
     {"plan": "Quantity by store means the total quantity per store: SUM(quantity) grouped by store.",
      "can_answer": True,
      "sql": "SELECT store, SUM(quantity) AS total_quantity FROM orders GROUP BY store"}),

    ("Which orders have no quantity recorded?",
     {"plan": "Return the orders themselves where quantity IS NULL, with their key and the asked column.",
      "can_answer": True,
      "sql": "SELECT order_id, store, quantity FROM orders WHERE quantity IS NULL"}),

    ("How many orders are missing a quantity, by store?",
     {"plan": "Count rows where quantity IS NULL, grouped by store as asked.",
      "can_answer": True,
      "sql": "SELECT store, COUNT(*) AS missing_quantity FROM orders "
             "WHERE quantity IS NULL GROUP BY store"}),

    ("What is the profit for each category?",
     {"plan": "Profit needs a cost column and orders has none; amount alone is revenue, so this cannot be answered.",
      "can_answer": False,
      "sql": ""}),
]


# ============================================================
# 2. MESSAGES
# ============================================================

# "the most released genre for every year", "top 3 products in each
# region", "best-selling item per month": the top value WITHIN each
# group. Small models answer these with a plain GROUP BY (every pair)
# or a subquery over the whole table (no rows), so they get a worked
# pattern. "Which region has the most sales" (no per / each / every)
# is a plain ranking and is not matched.
SUPERLATIVE = re.compile(
    r"\b(most|top|highest|largest|biggest|best|greatest|maximum|max|least|lowest|"
    r"fewest|smallest|worst|minimum|min|popular|common|frequent|leading|dominant)\b",
    re.IGNORECASE)
WITHIN_GROUP = re.compile(r"\b(per|each|every|within)\b", re.IGNORECASE)
SORT_ORDER = re.compile(
    r"\b(highest|lowest|largest|smallest|most|least|biggest|top)\s+"
    r"(first|last|to\s+\w+)\b", re.IGNORECASE)
TOP_COUNT = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
LOWEST = re.compile(r"\b(least|lowest|fewest|smallest|worst|minimum|min)\b", re.IGNORECASE)


def top_per_group(question):
    """
    {"count": n, "order": "DESC" | "ASC"} when the question asks for the
    top value within each group, else None.
    """

    # "highest first", "lowest to highest": a sort order, not a top value.
    question = SORT_ORDER.sub(" ", question)

    if not (SUPERLATIVE.search(question) and WITHIN_GROUP.search(question)):
        return None

    count = TOP_COUNT.search(question)

    return {"count": int(count.group(1)) if count else 1,
            "order": "ASC" if LOWEST.search(question) else "DESC"}


def top_per_group_hint(shape):
    """
    What the model is asked for under such a question: only one row per
    (group, item) with its number. The app keeps the top one(s) in each
    group itself (pipeline.keep_top_per_group), because small models
    write the ranking query unreliably.
    """

    return ("Note: this asks for the top value within each group. Do not rank, do "
            "not use LIMIT: return one row per group and item with its number, "
            "group first, number last, like:\n"
            "SELECT <group expression> AS grp, <item column> AS item, COUNT(*) AS n "
            "FROM <tables> GROUP BY 1, 2\n"
            "<group expression> is the column the question groups by (e.g. country). "
            "Only when the question names a period use per year strftime('%Y', <date>), "
            "per month strftime('%Y-%m', <date>) or per decade "
            "(CAST(strftime('%Y', <date>) AS INTEGER) / 10) * 10. "
            "Use SUM(<amount>) or AVG(<amount>) instead of COUNT(*) when the question "
            "is about an amount; MAX(<value>) for the highest single value. The app "
            "keeps the top item of each group.")


def user_message(schema_text, question, context=None):
    """Schema, optional conversation context, then the question."""

    shape = top_per_group(question)
    note = f"\n{top_per_group_hint(shape)}" if shape else ""

    if context:
        return f"Schema:\n{schema_text}\n\n{context}\n\nQuestion: {question}{note}"

    return f"Schema:\n{schema_text}\n\nQuestion: {question}{note}"


SAME_EXAMPLE_SCHEMA = "(the orders table shown above)"


def build_messages(schema_text, question, context=None):
    """
    System rules + few-shot pairs + the real schema, context and question.

    The example schema is written out in the first pair only; repeating
    it in all of them cost about 800 tokens of a 4096-token context.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for position, (example_question, example_answer) in enumerate(FEW_SHOT_EXAMPLES):
        example_schema = EXAMPLE_SCHEMA if position == 0 else SAME_EXAMPLE_SCHEMA
        messages.append({"role": "user",
                         "content": user_message(example_schema, example_question)})
        messages.append({"role": "assistant", "content": json.dumps(example_answer)})

    messages.append({"role": "user",
                     "content": user_message(schema_text, question, context)})

    return messages


def estimate_tokens(messages):
    """Rough prompt size in tokens (a few extra per message for its role)."""

    characters = sum(len(message["content"]) for message in messages)

    return characters // PROMPT_CHARS_PER_TOKEN + 4 * len(messages)


# ============================================================
# 3. GENERATING SQL
# ============================================================

def clean_sql(sql):
    """Strip whitespace, code fences and a trailing semicolon."""

    sql = sql.strip()

    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql[3:] if sql.lower().startswith("sql") else sql

    # A stray brace or fence left over from the JSON answer ("...'%Drama%'}").
    return sql.strip().strip("`").lstrip("{").rstrip("};").strip()


def generate_sql_from_messages(settings, messages):
    """
    One model call on a prepared conversation.

    Returns:
        {"plan", "can_answer", "sql", "seconds", "prompt_tokens",
         "output_tokens"}
    """

    answer, stats = chat_json(settings, messages, SQL_ANSWER_SCHEMA)

    return {"plan": answer["plan"].strip(),
            "can_answer": bool(answer["can_answer"]),
            "sql": clean_sql(answer["sql"]), **stats}


def generate_sql(settings, schema_text, question, context=None):
    """Ask the model for a query (single attempt, no checks)."""

    return generate_sql_from_messages(
        settings, build_messages(schema_text, question, context))
