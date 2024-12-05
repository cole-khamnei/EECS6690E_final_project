import os

# ----------------------------------------------------------------------------# 
# -------------             Directory And File Paths             -------------# 
# ----------------------------------------------------------------------------# 

CONSTANTS_FILE_PATH = os.path.realpath(__file__)
SRC_DIR_PATH = os.path.dirname(CONSTANTS_FILE_PATH)
PROJECT_DIR_PATH = os.path.dirname(SRC_DIR_PATH)
DATA_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "data")

THRYOID_CANCER_DATA_PATH = os.path.join(DATA_DIR_PATH, "thyroid_recurrence.csv")

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
