import json
import re
from collections import defaultdict

# Load JSON
with open("test_result_ff__uadfv.json", "r") as f:
    data = json.load(f)

# Counters
counts = defaultdict(int)

# Regex patterns
pattern_single = re.compile(r"^[^_]+$")       # Matches strings with no underscore
pattern_double = re.compile(r"^[^_]+_[^_]+$") # Matches exactly one underscore

# Classify each entry
for entry in data:
    name = entry["video_name"]
    label = entry["pred_label"].upper()

    if pattern_double.match(name):
        key = f"XXX_XXX_label_{label}"
    elif pattern_single.match(name):
        key = f"XXX_label_{label}"
    else:
        key = f"other_label_{label}"

    counts[key] += 1

# Print results
for k, v in counts.items():
    print(f"{k}: {v}")



# ------------------
# XXX_label_REAL: 220
# XXX_XXX_label_REAL: 211
# XXX_label_FAKE: 80
# XXX_XXX_label_FAKE: 89
# test_result_ff_org_f2f.json: 51.5% Acc

# ------------------------
# XXX_label_REAL: 154
# XXX_XXX_label_REAL: 175
# XXX_label_FAKE: 146
# XXX_XXX_label_FAKE: 125
# test_result_ff_uadfv_df.json: 53.5% Acc




