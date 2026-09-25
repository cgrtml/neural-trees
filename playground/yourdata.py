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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from neural_trees import SoftDecisionTreeRegressor

MAX_ROWS = 5000
MAX_COLS = 200
MAX_CLASSES = 20


def guess_target(df: pd.DataFrame) -> str:
    """A column named like a label, else the last column: that is where targets usually sit."""
    names = ("target", "label", "class", "y", "outcome", "diagnosis", "species", "survived", "churn",
             "default", "fraud", "result", "price", "saleprice", "amount", "score")
    for c in df.columns:
        if str(c).strip().lower() in names:
            return c
    return df.columns[-1]


def task_for(df: pd.DataFrame, target: str) -> str:
    """'classification' for a label-like column, 'regression' for a numeric one with many values."""
    s = df[target].dropna()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_CLASSES:
        return "regression"
    return "classification"


REGRESSORS = {
    "Soft Tree Regressor": dict(
        group="neural-trees", tag="The soft tree with a value per leaf",
        build=lambda: SoftDecisionTreeRegressor(depth=4, max_epochs=60, learning_rate=0.05, random_state=0),
        line="SoftDecisionTreeRegressor(depth=4, max_epochs=60, learning_rate=0.05, random_state=0)",
    ),
    "CART Regressor": dict(group="baseline", tag="One threshold per node, mean per leaf",
                           build=lambda: DecisionTreeRegressor(max_depth=5, random_state=0), line="DecisionTreeRegressor(max_depth=5)"),
    "Random Forest Regressor": dict(group="baseline", tag="Hundreds of trees averaging",
                                    build=lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1), line="RandomForestRegressor(n_estimators=200)"),
    "Ridge": dict(group="baseline", tag="A straight line through every feature",
                  build=lambda: Ridge(alpha=1.0), line="Ridge(alpha=1.0)"),
}


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
    """Return (ok, message) for using `target` as the label or the value to predict."""
    y = df[target].dropna()
    n = y.nunique()
    if task_for(df, target) == "regression":
        return True, (f"Regression: a number to predict. {n} distinct values, from {y.min():,.4g} to {y.max():,.4g}, "
                      f"median {y.median():,.4g}. Models are scored by R^2 (1 is perfect, 0 is predicting the mean).")
    if n < 2:
        return False, "The target has a single value; there is nothing to classify."
    if n > MAX_CLASSES:
        return False, (f"The target has {n} distinct text values. Above {MAX_CLASSES} classes this looks like an "
                       "identifier or free text, not something to predict.")
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
        if task_for(df, target) == "regression":
            data = data.sample(n=MAX_ROWS, random_state=seed)
            notes.append(f"Subsampled to {len(data)} rows (the hosted app caps at {MAX_ROWS}).")
        else:
            data = data.groupby(target, group_keys=False).apply(
                lambda g: g.sample(frac=MAX_ROWS / len(df), random_state=seed)
            )
            notes.append(f"Subsampled to {len(data)} rows (the hosted app caps at {MAX_ROWS}), keeping class proportions.")
    if task_for(df, target) == "regression":
        y = data[target].to_numpy(dtype=float)
        classes = []
    else:
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


def code_snippet(target: str, numeric, categorical, model_line: str, task: str = "classification", filename: str = "your_file.csv") -> str:
    """The same run as plain Python, commented line by line, for the user to take away."""
    cls = "SoftDecisionTree" if task == "classification" else "SoftDecisionTreeRegressor"
    step = "softdecisiontree" if task == "classification" else "softdecisiontreeregressor"
    y_line = f'df["{target}"].astype(str)' if task == "classification" else f'df["{target}"].astype(float)'
    reader = "pd.read_excel" if filename.lower().endswith((".xlsx", ".xls")) else "pd.read_csv"
    tail = (
        '# 5. Explain one row: the leaf it reached, the gates on the way, the smallest change that flips it.\n'
        'print(tree.explain(pre.transform(X.iloc[:1]))[0].to_text())\n'
        if task == "classification" else
        '# 5. Read the tree as rules with one value per leaf.\n'
        'print(tree.to_hard_tree().export_text())\n'
    )
    return f'''# pip install neural-trees scikit-learn pandas
# The same run as this page, on your own machine.
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from neural_trees import {cls}

# 1. Your table, and the column to predict.
df = {reader}("{filename}")
X, y = df.drop(columns=["{target}"]), {y_line}

# 2. The same preprocessing this page used: medians and scaling for numbers,
#    most-frequent value and one-hot encoding for categories, all fitted inside each fold.
numeric = {list(map(str, numeric))}
categorical = {list(map(str, categorical))}
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
])

# 3. The model, scored with 5-fold cross-validation ({"accuracy" if task == "classification" else "R^2"}).
model = make_pipeline(pre, {model_line})
print(cross_val_score(model, X, y, cv=5).mean())

# 4. Fit on everything and keep the tree.
model.fit(X, y)
tree = model.named_steps["{step}"]
{tail}
# 6. Save it: model.json predicts with numpy alone, no torch needed where it is served.
tree.to_numpy().to_json("model.json")
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
