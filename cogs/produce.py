"""Produce-family grouping for pack types (ProductIDs).

A *produce family* is the fruit a buyer quotes a single raw price for
(e.g. "Cherry"), which then fans out across several pack types
("Cherry 10 x 500g", "Cherry 5kg Loose", …). The mapping is stored in
`product_groups.csv`; these pure helpers provide a heuristic default for new
products and reconcile the stored mapping against the live product list, so the
Season Pricing page never has to trust the file to be complete or current.

UI-free so it can be unit-tested without Streamlit.
"""
import re

import pandas as pd

GROUP_COLS = ["ProductID", "Produce", "AirEligible"]

# Known fruit keywords, most-specific first so "UFO Peach" / "Green Plum" win
# over "Peach" / "Plum", and multi-word fruits resolve before their bare form.
_KNOWN_PRODUCE = [
    "UFO Nectarine",
    "UFO Peach",
    "Green Plum",
    "Watermelon (Seedless)",
    "Watermelon (Seeded)",
    "Watermelon",
    "Grapefruit",
    "Nectarine",
    "Strawberry",
    "Pomegranate",
    "Apricot",
    "Cherry",
    "Peach",
    "Lemon",
    "Plum",
    "Pear",
    "Fig",
]


def _coerce_bool(value) -> bool:
    """Parse a CSV cell or edited value into a bool. Unknown/blank → True
    (most pack types are air-eligible; the seed marks the exceptions)."""
    if isinstance(value, bool):  # must precede int — bool is a subclass of int
        return value
    if value is None:
        return True
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return bool(value)
    s = str(value).strip().lower()
    return s not in {"false", "0", "no", "n", "f"}


def default_produce(product_id: str) -> str:
    """Best-guess produce family for an unmapped ProductID.

    Matches a known fruit keyword anywhere in the name, so both
    "14kg Open Top Grapefruit" (fruit last) and "Black Fig 700g" (fruit first)
    resolve. Falls back to the first non-size token so a brand-new product still
    groups under *something* sensible.
    """
    name = str(product_id).strip()
    if not name:
        return ""
    low = name.lower()
    for produce in _KNOWN_PRODUCE:
        if produce.lower() in low:
            return produce
    for tok in name.split():
        if not re.match(r"^\d", tok):  # skip leading size tokens like "14kg"
            return tok
    return name


def reconcile_groups(groups_df: pd.DataFrame, product_ids) -> pd.DataFrame:
    """Align the stored grouping with the live product list.

    - Appends any ProductID in ``product_ids`` missing from ``groups_df``,
      seeding Produce via :func:`default_produce` and ``AirEligible=True``.
    - Drops rows whose ProductID is no longer a live product.
    - Coerces ``AirEligible`` to a clean bool and fills blank Produce cells.

    Returns a fresh frame with columns :data:`GROUP_COLS`, one row per live
    product, ordered to match ``product_ids``.
    """
    live = [str(p).strip() for p in product_ids if str(p).strip()]
    live_set = set(live)

    existing: dict[str, tuple[str, bool]] = {}
    if groups_df is not None and not groups_df.empty:
        df = groups_df.copy()
        df["ProductID"] = df["ProductID"].astype(str).str.strip()
        produce_col = df["Produce"].astype(str).str.strip() if "Produce" in df else ""
        df["Produce"] = produce_col
        eligible_col = df["AirEligible"] if "AirEligible" in df else True
        df["AirEligible"] = (
            eligible_col.apply(_coerce_bool)
            if hasattr(eligible_col, "apply")
            else True
        )
        for _, row in df.iterrows():  # last value wins on duplicate ProductIDs
            if row["ProductID"] in live_set:
                existing[row["ProductID"]] = (row["Produce"], bool(row["AirEligible"]))

    records = []
    for pid in live:
        if pid in existing:
            produce, eligible = existing[pid]
            produce = produce or default_produce(pid)
        else:
            produce, eligible = default_produce(pid), True
        records.append(
            {"ProductID": pid, "Produce": produce, "AirEligible": bool(eligible)}
        )

    return pd.DataFrame(records, columns=GROUP_COLS)
