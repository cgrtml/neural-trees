# On macOS, torch and XGBoost each load an OpenMP runtime; in one process the
# second one to start threads can segfault (seen with xgboost 3.4.1 and torch
# 2.14.1, in either import order). One OpenMP thread avoids it. This must be set
# before either library is imported, so it lives here.
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
