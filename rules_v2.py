"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).
"""

import rules
import cv2
import numpy as np

#: Import the original rules and append your new one.
# Need to reweight the previous rules, so I am coping them and changing their weights
RULES = ("rule1_b", "rule2_b", "rule3_b", "rule4_edges")

def _reweight(out, new_out_of):
     out["score"] = round(out["score"] * new_out_of / out["out_of"])
     out["out_of"] = new_out_of
     return out

def rule1_b(target, input_path):
     return _reweight(rules.rule1_metadata(target, input_path), 25)

def rule2_b(target, input_path):
     return _reweight(rules.rule2_histogram(target, input_path), 25)

def rule3_b(target, input_path):
     return _reweight(rules.rule3_template(target, input_path), 40)

# Noticed errors of rules 1-3 on hard set
# 40 matched, 20 wrong
# Common pattern appears to be error in resizing and compressing  -> aim for rule 4 to address this
# Like for fixing rule 3 in rules.py, I googled the recommendations on the instructions and set on multi-scale template matching
# I used the template from this guide to match with the rule 4 template below: https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

def rule4_edges(target, input_path):
    """TODO: replace with your Phase 2 rule that fixes the V1 weakness.

    Starter returns a no-op so the pipeline still runs before you implement it.
    Parity with rules.py rule dict: rule/name/fired/score/out_of/note/metric.
    """
    src = rules._gray(target["path"])
    sus = rules._gray(input_path)
    s = min(1.0, 512 / max(src.shape))
    src = cv2.resize(src, None, fx=s, fy=s)
    best = 0.0
    for k in np.linspace(0.5, 1.5, 21):
        t = cv2.resize(sus, None, fx=s * k, fy=s * k)
        if t.shape[0] <= src.shape[0] and t.shape[1] <= src.shape[1]:
            best = max(best, float(cv2.matchTemplate(src, t, cv2.TM_CCOEFF_NORMED).max()))
    return {
        "rule": 4,
        "name": "Edges",
        "fired": best >= 0.75,
        "score": round(10* best) if best >= 0.75 else 0,
        "out_of": 10,
        "note": "Multi-scale template matching",
        "metric": 0.0,
    }
