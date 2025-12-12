import json
import re
from collections import defaultdict

# Load JSON
with open("4. Celeb-Org.json", "r") as f:
    data = json.load(f)

# Counters
counts = defaultdict(int)

# Regex patterns
pattern_id_pair = re.compile(r"^id\d+_id\d+_\d+$")
pattern_digits = re.compile(r"^\d+$")

# Classify each entry
for entry in data:
    name = entry["video_name"]
    label = entry["pred_label"].upper()

    if pattern_id_pair.match(name):
        key = f"id_id_label_{label}"
    elif pattern_digits.match(name):
        key = f"digits_label_{label}"
    else:
        key = f"other_label_{label}"

    counts[key] += 1

# Print results
for k, v in counts.items():
    print(f"{k}: {v}")


# --------------------------------

# id_id_label_REAL: 193
# id_id_label_FAKE: 107
# digits_label_FAKE: 147
# digits_label_REAL: 153

# 56.67%: Celeb-DF: test_result.json: UADFV Model


# --------------------------------

# id_id_label_REAL: 222
# id_id_label_FAKE: 78
# digits_label_REAL: 187
# digits_label_FAKE: 113

# 55.83%: Celeb-DF: test_result_org.json: Original Model

