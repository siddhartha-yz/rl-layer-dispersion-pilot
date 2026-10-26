"""Independently verify rewards, split overlap, and the question-blind control."""
from collections import Counter, defaultdict
import operator
from rl_layer_pilot import ROOT, make_data, save_json

ops = {"+": operator.add, "-": operator.sub, "*": operator.mul}
seen = set()
audit = {}
for split, rows in make_data().items():
    blocks = defaultdict(list)
    for r in rows:
        assert r["options"][r["label"]] == ops[r["op"]](r["a"], r["b"])
        assert len(set(r["options"])) == 4
        key = (r["op"], *sorted((r["a"], r["b"]))) if r["op"] != "-" else (r["op"], r["a"], r["b"])
        assert key not in seen
        seen.add(key)
        blocks[r["block"]].append(r)
    for block in blocks.values():
        assert len(block) == 4 and {r["label"] for r in block} == {0, 1, 2, 3}
        assert all(r["options"] == block[0]["options"] and r["op"] == block[0]["op"] for r in block)
    audit[split] = dict(questions=len(rows), blocks=len(blocks),
                        operations=dict(Counter(r["op"] for r in rows)),
                        any_options_only_rule_accuracy=.25)
save_json(ROOT / "runs/data_audit.json", audit)
print(audit)
