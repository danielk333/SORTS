import numpy as np


max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(np.iinfo(np.int64).min + 1, "us")  # +1 is needed, otherwise it will be NaT; fmt: skip;
