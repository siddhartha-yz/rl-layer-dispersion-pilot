"""Checks directed at the pilot's actual masking and RL-gradient risks."""
import torch
from rl_layer_pilot import Policy, make_data, save_json, ROOT

torch.set_num_threads(4)
torch.manual_seed(0)
p = Policy()
rows = make_data()["train"][:4]
batch, labels = p.batch(rows)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    native_logits = p.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1]
    native = native_logits[:, p.action_ids].float().log_softmax(-1)
    manual = p.logp(batch)
checks = {"conditional_logits_match_native": bool(torch.allclose(native, manual, atol=1e-6)),
          "no_pretrained_parameter_trainable": all(not x.requires_grad for n, x in p.model.named_parameters()
                                                    if not n.endswith((".A", ".B")))}
with torch.no_grad():
    for layer in p.layers:
        for m in layer:
            m.B.normal_(std=.003)
p.set_gates([0.] * p.n_layers)
with torch.no_grad():
    checks["all_increments_off_restores_pretrained_policy"] = bool(torch.allclose(p.logp(batch), native, atol=1e-6))
active = [k for k in range(p.n_layers) if k % 4 != 0]
p.set_gates([float(k in active) for k in range(p.n_layers)])
p.set_trainable(active)
loss = -p.logp(batch).gather(1, labels[:, None]).mean()
loss.backward()
checks["dropped_layers_have_no_adapter_gradients"] = all(m.A.grad is None and m.B.grad is None
    for k, layer in enumerate(p.layers) if k not in active for m in layer)
checks["every_active_layer_receives_gradient"] = all(any(m.B.grad is not None and m.B.grad.abs().sum() > 0 for m in layer)
    for k, layer in enumerate(p.layers) if k in active)
checks["pretrained_parameters_have_no_gradients"] = all(x.grad is None for n, x in p.model.named_parameters()
    if not n.endswith((".A", ".B")))
# A rollout mask must not change between the sampled policy and its logprob.
with torch.no_grad():
    x, y = p.logp(batch), p.logp(batch)
    checks["fixed_mask_logprobs_reproducible"] = bool(torch.equal(x, y))
save_json(ROOT / "runs/mechanism_checks.json", checks)
print(checks, flush=True)
if not all(checks.values()):
    raise RuntimeError("Pilot mechanism verification failed")
