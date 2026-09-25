"""
Turning an uploaded table into something the models can train on.

Kept apart from the registry because it is the one place the playground
touches data it did not ship with: the rules for what a target column can
be, how categorical columns are encoded, what happens to missing values and
how large a table the hosted app will take are all here, and each one is
stated to the user on the page.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_ROWS = 5000
MAX_COLS = 200
MAX_CLASSES = 20


def guess_target(df: pd.DataFrame) -> str:
    """The last column unless a column is named like a label."""
    names = ("target", "label", "class", "y", "outcome", "diagnosis", "species", "survived", "churn",
             "default", "fraud", "result", "status", "category", "type")
    for c in df.columns:
        if str(c).strip().lower() in names:
            return c
    # otherwise the last column with 2 to MAX_CLASSES distinct values, else the last column
    for c in reversed(df.columns):
        if 2 <= df[c].nunique(dropna=True) <= MAX_CLASSES:
            return c
    return df.columns[-1]


def describe(df: pd.DataFrame, target: str):
    """What the page will do with each column, as rows for a table."""
    rows = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) == len(s.dropna()) and len(s) > 20 and pd.api.types.is_integer_dtype(s):
                kind, action = "integer id", "dropped: a different value on every row, so it identifies rows rather than describing them"
            elif s.nunique(dropna=True) <= 1:
                kind, action = "constant", "dropped: the same value on every row"
            else:
                kind, action = "numeric", "standardised on each training fold; missing values filled with the median"
        else:
            n = s.nunique(dropna=True)
            if n > 50:
                kind, action = "text, many values", "dropped: too many distinct values to encode"
            else:
                kind, action = f"categorical, {n} values", "one-hot encoded"
        rows.append({"column": str(c), "type": kind, "missing": int(s.isna().sum()), "what happens": action})
    return pd.DataFrame(rows).set_index("column")


def check_target(df: pd.DataFrame, target: str):
    """Return (ok, message) for using `target` as the classification label."""
    y = df[target].dropna()
    n = y.nunique()
    if n < 2:
        return False, "The target has a single value; there is nothing to classify."
    if n > MAX_CLASSES:
        return False, (f"The target has {n} distinct values. Above {MAX_CLASSES} this is a regression "
                       "problem or an identifier, and this page does classification only.")
    counts = y.value_counts()
    if counts.min() < 5:
        return False, (f"The rarest class ({counts.idxmin()!r}) has {counts.min()} rows; five-fold "
                       "cross-validation needs at least 5 of each.")
    return True, f"{n} classes; the rarest has {counts.min()} rows, the commonest {counts.max()}."


def prepare(df: pd.DataFrame, target: str, seed: int = 0):
    """
    Split the table into X (a DataFrame of usable columns) and y (encoded labels),
    subsampling to MAX_ROWS with stratification. Returns X, y, class names,
    the list of numeric and categorical columns, and a note about what was dropped.
    """
    data = df.dropna(subset=[target]).copy()
    notes = []
    if len(data) > MAX_ROWS:
        data = data.groupby(target, group_keys=False).apply(
            lambda g: g.sample(frac=MAX_ROWS / len(df), random_state=seed)
        )
        notes.append(f"Subsampled to {len(data)} rows (the hosted app caps at {MAX_ROWS}), keeping class proportions.")
    y_raw = data[target].astype(str)
    classes = sorted(y_raw.unique())
    y = np.array([classes.index(v) for v in y_raw])
    numeric, categorical, dropped = [], [], []
    for c in data.columns:
        if c == target:
            continue
        s = data[c]
        if pd.api.types.is_numeric_dtype(s):
            n = s.nunique(dropna=True)
            is_id = n == len(s.dropna()) and len(s) > 20 and pd.api.types.is_integer_dtype(s)
            if n > 1 and not is_id:
                numeric.append(c)
            else:
                dropped.append(c)
        elif s.nunique(dropna=True) <= 50:
            categorical.append(c)
        else:
            dropped.append(c)
    if len(numeric) + len(categorical) > MAX_COLS:
        numeric = numeric[: max(0, MAX_COLS - len(categorical))]
        notes.append(f"Kept the first {MAX_COLS} usable columns.")
    if dropped:
        notes.append("Dropped: " + ", ".join(str(c) for c in dropped[:8]) + (" ..." if len(dropped) > 8 else "") + " (constant, an integer id, or text with too many distinct values).")
    X = data[numeric + categorical]
    return X, y, classes, numeric, categorical, notes


def preprocessor(numeric, categorical):
    """Median-impute and standardise numerics, one-hot categoricals; fitted inside every fold."""
    parts = []
    if numeric:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical))
    return ColumnTransformer(parts, remainder="drop")


def feature_names_after(pre: ColumnTransformer):
    """Column names after preprocessing, readable in the explanations."""
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names.extend(str(c) for c in cols)
        elif name == "cat":
            enc = trans.named_steps["onehotencoder"]
            for c, cats in zip(cols, enc.categories_):
                names.extend(f"{c}={v}" for v in cats)
    return names


def code_snippet(target: str, numeric, categorical, model_line: str) -> str:
    """The same pipeline as plain Python, for the user to take away."""
    return f'''import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from neural_trees import SoftDecisionTree, combined_5x2cv_f_test

df = pd.read_csv("your_file.csv")
X, y = df.drop(columns=["{target}"]), df["{target}"].astype(str)
numeric = {list(map(str, numeric))}
categorical = {list(map(str, categorical))}
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
])
model = make_pipeline(pre, {model_line})
print(cross_val_score(model, X, y, cv=5).mean())

model.fit(X, y)
tree = model.named_steps["softdecisiontree"]
print(tree.explain(pre.transform(X.iloc[:1]), feature_names=None)[0].to_text())
tree.to_numpy().to_json("model.json")          # the same model, no torch needed to predict
'''


def read_table(raw: bytes, filename: str = "") -> pd.DataFrame:
    """An uploaded CSV or Excel file as a DataFrame; Excel takes the first sheet."""
    import io

    if filename.lower().endswith((".xlsx", ".xls")) or raw[:4] == b"PK\x03\x04":
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
        if df.shape[1] < 2 or len(df) < 10:
            raise ValueError(f"the first sheet has {df.shape[1]} columns and {len(df)} rows; at least 2 columns and 10 rows are needed")
        return df
    return read_csv(raw)


def read_csv(raw: bytes) -> pd.DataFrame:
    """
    Read an uploaded CSV without asking about separators or encodings: the
    separator is sniffed, UTF-8 is tried first and Latin-1 second, and a file
    that parses to a single column is reported rather than accepted.
    """
    import io

    head = raw[:4096]
    printable = sum(b in b"\r\n\t" or 32 <= b < 127 or b >= 128 for b in head) / max(1, len(head))
    if printable < 0.95 or b"\x00" in head:
        raise ValueError("this does not look like a text file")
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except Exception as e:  # noqa: BLE001 - reported to the user
            last = e
            continue
        if df.shape[1] < 2:
            last = ValueError("the file parsed to a single column; is it comma, semicolon or tab separated?")
            continue
        if len(df) < 10:
            last = ValueError(f"only {len(df)} rows; five-fold cross-validation needs at least 10")
            continue
        return df
    raise ValueError(str(last))
