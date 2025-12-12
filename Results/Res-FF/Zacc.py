import json
import re
from collections import defaultdict

# Load JSON
with open("test_result_ff_org.json", "r") as f:
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



# ---------------------- 
# XXX_label_REAL: 222
# XXX_XXX_label_REAL: 210
# XXX_label_FAKE: 78
# XXX_XXX_label_FAKE: 90


# Acc: 50% : Org-Model---- test_result_ff_org.json
# ----------------------




