"""Secondary check for non-inverted dropout's train/deploy scale mismatch.

Select a single deployment scale on seed-0 tuning validation only, then apply
it to all three held-out training seeds. Primary unscaled results are retained.
"""
import json
import numpy as np
import torch
from rl_layer_pilot import ROOT, Policy, make_data, metrics, evaluate, save_json

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
data = make_data()
p = Policy()
chosen_lr = json.loads((ROOT / "runs/selected_lrs.json").read_text())["delta_drop"]
tune = ROOT / "runs/tune" / f"delta_drop_{chosen_lr:g}"
p.restore(torch.load(tune / "adapter.pt", map_location="cpu", weights_only=True))
validation = {}
for scale in [1., .75]:
    p.set_gates([scale] * p.n_layers)
    validation[str(scale)] = metrics(p.predict(data["validation"]), data["validation"])
scale = max([1., .75], key=lambda q: validation[str(q)]["expected_reward"])
scores = []
for seed in [11, 22, 33]:
    file = ROOT / "runs/delta_drop" / f"seed_{seed}" / "adapter.pt"
    p.restore(torch.load(file, map_location="cpu", weights_only=True))
    p.set_gates([scale] * p.n_layers)
    scores.append(dict(seed=seed, scale=scale, scores=evaluate(p, data)))
save_json(ROOT / "runs/deployment_scale_diagnostic.json", dict(
    status="Secondary analysis; primary training and unscaled scores unchanged.",
    selection="Seed-0 tuning validation expected reward, one common scale for all three main seeds.",
    validation=validation, selected_scale=scale, runs=scores))
print(json.dumps(dict(validation=validation, selected_scale=scale,
                     id=[s["scores"]["test_id"]["accuracy"] for s in scores],
                     ood=[np.mean([s["scores"][k]["accuracy"] for k in ["test_format", "test_large"]]) for s in scores]), indent=2))
