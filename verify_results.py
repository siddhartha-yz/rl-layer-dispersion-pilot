"""Reload saved adapters and independently rescore every reported ID result."""
import json
import torch
from rl_layer_pilot import ROOT, Policy, make_data, metrics, save_json

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
p = Policy()
rows = make_data()["test_id"]
checks = []
for file in sorted((ROOT / "runs").glob("*/seed_*/result.json")):
    recorded = json.loads(file.read_text())
    state = torch.load(file.parent / "adapter.pt", map_location="cpu", weights_only=True)
    p.restore(state)
    p.set_gates([1.] * p.n_layers)
    rescored = metrics(p.predict(rows), rows)
    before = recorded["scores"]["test_id"]
    ok = abs(rescored["accuracy"] - before["accuracy"]) < 1e-8 and abs(rescored["expected_reward"] - before["expected_reward"]) < 1e-6
    checks.append(dict(arm=recorded["arm"], seed=recorded["seed"], matched=ok,
                       rescored=rescored, recorded=before))
    print(recorded["arm"], recorded["seed"], ok, flush=True)
save_json(ROOT / "runs/reloaded_checkpoint_checks.json", checks)
assert len(checks) == 12 and all(c["matched"] for c in checks)
