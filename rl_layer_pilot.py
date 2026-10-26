"""Reproducible, one-action RLVR pilot on a real pretrained Qwen policy.

No supervised loss is used: sample eight A/B/C/D actions per question, verify
their rewards, and use within-group standardized advantages. The action space
is deliberately restricted to four answer tokens, so this is a contextual
bandit, not a chain-of-thought GRPO reproduction.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/qwen2.5-0.5b-instruct"
LETTERS = "ABCD"


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def make_data():
    """Four distinct questions share the same ordered options, one per answer.

    Every options-only rule is exactly 25% on complete blocks, even if it knows
    the operation and number distribution. Arithmetic identities never cross
    splits. Blocks are also the units used by the uncertainty calculation.
    """
    rng = random.Random(20261024)
    next_block = 0

    def pool(low, high):
        pools = {op: {} for op in ["+", "-", "*"]}
        for op in pools:
            width = {"+": 20, "-": 10, "*": 100}[op]
            for a in range(low, high + 1):
                for b in range(low, high + 1):
                    if op in ["+", "*"] and a > b:
                        continue
                    if op == "-" and a < b:
                        continue
                    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
                    pools[op].setdefault(answer // width, {}).setdefault(answer, []).append((a, b))
        return pools

    def split(n, pools, style):
        nonlocal next_block
        assert n % 4 == 0
        rows = []
        while len(rows) < n:
            available = {op: [bucket for bucket, answers in buckets.items() if len(answers) >= 4]
                         for op, buckets in pools.items()}
            op = rng.choice([op for op, bins in available.items() if bins])
            bucket = rng.choice(available[op])
            answers = pools[op][bucket]
            options = rng.sample(list(answers), 4)
            block = next_block
            next_block += 1
            for label, answer in enumerate(options):
                cases = answers[answer]
                a, b = cases.pop(rng.randrange(len(cases)))
                if not cases:
                    del answers[answer]
                if style == "plain":
                    question = f"Calculate {a} {op} {b}."
                    choices = "\n".join(f"{c}) {v}" for c, v in zip(LETTERS, options))
                    prompt = f"Choose the correct answer. Reply with A, B, C, or D only.\n{question}\n{choices}\nAnswer:"
                else:
                    word = {"+": "plus", "-": "minus", "*": "times"}[op]
                    question = f"What is {a} {word} {b}?"
                    choices = "; ".join(f"({c}) {v}" for c, v in zip(LETTERS, options))
                    prompt = f"{question}\nOptions: {choices}.\nSelect one letter (A/B/C/D):"
                rows.append(dict(prompt=prompt, label=label, a=a, b=b, op=op,
                                 answer=answer, options=options[:], block=block,
                                 answer_rank=sorted(options).index(answer)))
        rng.shuffle(rows)
        return rows
    small = pool(2, 49)
    return dict(train=split(1536, small, "plain"),
                validation=split(192, small, "plain"),
                test_id=split(384, small, "plain"),
                test_format=split(384, small, "words"),
                test_large=split(384, pool(50, 99), "plain"))


class DeltaLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device="cuda"))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device="cuda"))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = 2.0  # alpha = 2 * rank
        self.gate = 1.0

    def forward(self, x):
        y = self.base(x)
        if self.gate == 0:
            return y
        delta = F.linear(F.linear(x, self.A), self.B)
        return y + delta.to(y.dtype) * (self.scale * self.gate)


class Policy:
    def __init__(self, rank=8):
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        self.model.requires_grad_(False)
        self.action_ids = torch.tensor([self.tok.encode(c, add_special_tokens=False)[0]
                                        for c in LETTERS], device="cuda")
        assert all(len(self.tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
        self.layers = []
        for layer in self.model.model.layers:
            modules = []
            for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                  (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
                for name in names:
                    wrapped = DeltaLinear(getattr(parent, name), rank)
                    setattr(parent, name, wrapped)
                    modules.append(wrapped)
            self.layers.append(modules)
        self.n_layers = len(self.layers)
        self.initial = self.state()

    def state(self):
        return {f"{k}.{i}.{name}": getattr(m, name).detach().cpu().clone()
                for k, layer in enumerate(self.layers) for i, m in enumerate(layer)
                for name in ["A", "B"]}

    def restore(self, state):
        with torch.no_grad():
            for k, layer in enumerate(self.layers):
                for i, m in enumerate(layer):
                    for name in ["A", "B"]:
                        getattr(m, name).copy_(state[f"{k}.{i}.{name}"])

    def reset(self, seed):
        # Paired initial A matrices and zero B for every treatment within a seed.
        torch.manual_seed(seed)
        with torch.no_grad():
            for layer in self.layers:
                for m in layer:
                    nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
                    m.B.zero_()
        self.set_gates([1.] * self.n_layers)
        self.set_trainable(range(self.n_layers))

    def params(self):
        return [p for layer in self.layers for m in layer for p in [m.A, m.B]]

    def set_trainable(self, active):
        active = set(active)
        for k, layer in enumerate(self.layers):
            for m in layer:
                m.A.requires_grad_(k in active)
                m.B.requires_grad_(k in active)

    def set_gates(self, gates):
        for gate, layer in zip(gates, self.layers):
            for m in layer:
                m.gate = float(gate)

    def batch(self, rows):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
                 for r in rows]
        t = self.tok(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        t = {k: v.to("cuda") for k, v in t.items()}
        t["position_ids"] = (t["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        y = torch.tensor([r["label"] for r in rows], device="cuda")
        return t, y

    def logp(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model.model(**batch, use_cache=False).last_hidden_state[:, -1]
            w = self.model.lm_head.weight.index_select(0, self.action_ids)
            logits = F.linear(h, w).float()
        return logits.log_softmax(-1)

    @torch.no_grad()
    def predict(self, rows, batch_size=32):
        ps = []
        for start in range(0, len(rows), batch_size):
            t, _ = self.batch(rows[start:start + batch_size])
            ps.append(self.logp(t).exp().cpu())
        return torch.cat(ps).numpy()


def metrics(probs, rows):
    y = np.array([r["label"] for r in rows])
    return dict(accuracy=float((probs.argmax(-1) == y).mean()),
                expected_reward=float(probs[np.arange(len(y)), y].mean()),
                entropy=float((-probs * np.log(probs.clip(1e-12))).sum(-1).mean()),
                by_operation={op: float((probs.argmax(-1)[[r["op"] == op for r in rows]] ==
                                         y[[r["op"] == op for r in rows]]).mean())
                              for op in ["+", "-", "*"] if any(r["op"] == op for r in rows)})


def evaluate(policy, data):
    result = {name: metrics(policy.predict(rows), rows)
              for name, rows in data.items() if name != "train"}
    # Same held-out options/labels and operator, but an unrelated arithmetic
    # question. High accuracy here would expose an answer-option shortcut.
    original = data["test_id"]
    shuffled = [dict(r) for r in original]
    for op in ["+", "-", "*"]:
        ids = [i for i, r in enumerate(original) if r["op"] == op]
        for i, j in zip(ids, ids[1:] + ids[:1]):
            r, other = original[i], original[j]
            old = f"Calculate {r['a']} {op} {r['b']}."
            new = f"Calculate {other['a']} {op} {other['b']}."
            shuffled[i]["prompt"] = r["prompt"].replace(old, new)
    result["test_question_shuffle"] = metrics(policy.predict(shuffled), shuffled)
    return result


def ablation_rows(data):
    rows = data["test_id"]
    blocks = set(sorted({r["block"] for r in rows})[:32])
    return [r for r in rows if r["block"] in blocks]


def ablate(policy, rows):
    """Remove only the learned increment, never a pretrained Transformer layer."""
    baseline = policy.predict(rows)
    full = metrics(baseline, rows)
    findings = []
    for k in range(policy.n_layers):
        gates = np.ones(policy.n_layers)
        gates[k] = 0
        policy.set_gates(gates)
        probs = policy.predict(rows)
        m = metrics(probs, rows)
        m.update(layer=k,
                 reward_drop=full["expected_reward"] - m["expected_reward"],
                 accuracy_drop=full["accuracy"] - m["accuracy"],
                 kl_from_full=float((baseline * (np.log(baseline.clip(1e-12)) -
                                                np.log(probs.clip(1e-12)))).sum(-1).mean()))
        findings.append(m)
    policy.set_gates([1.] * policy.n_layers)
    positive = np.maximum([r["reward_drop"] for r in findings], 0)
    total = float(positive.sum())
    effective = float(total ** 2 / np.square(positive).sum()) if total else 0.
    return dict(full=full, layers=findings, effective_positive_layers=effective,
                max_reward_drop=max(r["reward_drop"] for r in findings),
                warning="Single-layer removals are non-additive; this is a sensitivity proxy.")


def train(policy, data, arm, seed, lr, steps, batch_size, outdir, do_ablate=True):
    policy.reset(seed)
    optimizer = torch.optim.AdamW(policy.params(), lr=lr, weight_decay=0)
    rng = np.random.default_rng(seed + 1000)
    mask_rng = np.random.default_rng(seed + 2000)
    sample_rng = torch.Generator(device="cuda").manual_seed(seed + 3000)
    middle = list(range(policy.n_layers // 2 - 3, policy.n_layers // 2 + 3))
    all_layers = list(range(policy.n_layers))
    active_counts = np.zeros(policy.n_layers, dtype=int)
    if arm == "middle":
        policy.set_trainable(middle)
    log = []
    begin = time.perf_counter()
    for step in range(steps):
        chosen = rng.choice(len(data["train"]), batch_size, replace=False)
        rows = [data["train"][int(i)] for i in chosen]
        gates = np.ones(policy.n_layers)
        active = all_layers
        if arm == "middle":
            active = middle
        elif arm == "rotate":
            # Freeze parameter updates, but preserve the learned forward function.
            absent = mask_rng.choice(policy.n_layers, 6, replace=False)
            active = [k for k in all_layers if k not in absent]
            policy.set_trainable(active)
        elif arm == "delta_drop":
            # Exactly six whole-layer adapter increments absent per rollout batch.
            absent = mask_rng.choice(policy.n_layers, 6, replace=False)
            gates[absent] = 0
            active = [k for k in all_layers if gates[k]]
            policy.set_trainable(active)
        policy.set_gates(gates)
        active_counts[active] += 1
        batch, y = policy.batch(rows)
        optimizer.zero_grad(set_to_none=True)
        logp = policy.logp(batch)
        with torch.no_grad():
            actions = torch.multinomial(logp.exp(), 8, replacement=True, generator=sample_rng)
            reward = (actions == y[:, None]).float()
            advantage = (reward - reward.mean(-1, keepdim=True)) / reward.std(-1, keepdim=True, unbiased=False).clamp(min=1e-6)
        selected = logp.gather(1, actions)
        # Exactly on-policy, one update: ratio == 1 before update. Gradient is
        # the one-update GRPO surrogate; clipping has no effect at this point.
        ratio = (selected - selected.detach()).exp()
        kl = (logp.exp() * (logp - policy.reference_logp[chosen])).sum(-1).mean()
        loss = -(ratio * advantage).mean() + .1 * kl
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.params(), 1.)
        optimizer.step()
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite optimization state")
        if (step + 1) % 20 == 0 or step == 0:
            item = dict(step=step + 1, sampled_reward=reward.mean().item(),
                        expected_reward=logp.exp().gather(1, y[:, None]).mean().item(),
                        gradient_norm=float(grad_norm), seconds=time.perf_counter() - begin)
            log.append(item)
            print(json.dumps(dict(arm=arm, seed=seed, lr=lr, **item)), flush=True)
    policy.set_gates([1.] * policy.n_layers)
    scores = evaluate(policy, data) if do_ablate else {
        "validation": metrics(policy.predict(data["validation"]), data["validation"])}
    result = dict(arm=arm, seed=seed, lr=lr, steps=steps, batch_size=batch_size,
                  group_size=8, sampled_actions=steps * batch_size * 8,
                  train_seconds=time.perf_counter() - begin, scores=scores,
                  active_counts=active_counts.tolist(), log=log)
    if do_ablate:
        result["ablation"] = ablate(policy, ablation_rows(data))
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "result.json", result)
    torch.save(policy.state(), outdir / "adapter.pt")
    prediction_splits = [k for k in data if k != "train"] if do_ablate else ["validation"]
    np.savez_compressed(outdir / "predictions.npz",
                        **{k: policy.predict(data[k]) for k in prediction_splits})
    print(json.dumps(dict(done=str(outdir), scores=scores)), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "pilot", "sweep"], default="probe")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--arms", nargs="+", default=["all", "middle", "rotate", "delta_drop"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-5, 1e-4])
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = make_data()
    save_json(ROOT / "data/arithmetic.json", data)
    torch.manual_seed(0)
    policy = Policy()
    reference_path = args.out / "reference_logp.npy"
    if reference_path.exists():
        reference_logp = np.load(reference_path)
    else:
        reference_logp = np.log(policy.predict(data["train"]).clip(1e-12))
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(reference_path, reference_logp)
    policy.reference_logp = torch.tensor(reference_logp, device="cuda")
    environment = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                       layers=policy.n_layers, trainable_parameters=sum(p.numel() for p in policy.params()),
                       config=vars(args) | {"out": str(args.out)},
                       dataset_sha256=hashlib.sha256((ROOT / "data/arithmetic.json").read_bytes()).hexdigest())
    save_json(args.out / "environment.json", environment)
    baseline_path = args.out / "baseline.json"
    if not baseline_path.exists():
        baseline = evaluate(policy, data)
        save_json(baseline_path, baseline)
        print(json.dumps(dict(baseline=baseline)), flush=True)
    if not (args.out / "baseline_ablation_subset.json").exists():
        save_json(args.out / "baseline_ablation_subset.json",
                  metrics(policy.predict(ablation_rows(data)), ablation_rows(data)))
    if args.mode == "probe":
        train(policy, data, "all", 0, args.lr, args.steps, args.batch_size, args.out / "probe", False)
        return
    if args.mode == "sweep":
        for arm in args.arms:
            for lr in args.lrs:
                dest = args.out / f"tune/{arm}_{lr:g}"
                if not (dest / "result.json").exists():
                    train(policy, data, arm, 0, lr, args.steps, args.batch_size, dest, False)
        return
    chosen_lrs = {}
    for arm in args.arms:
        candidates = []
        for p in (args.out / "tune").glob(f"{arm}_*/result.json"):
            d = json.loads(p.read_text())
            candidates.append((d["scores"]["validation"]["expected_reward"], d["lr"]))
        chosen_lrs[arm] = max(candidates)[1] if candidates else args.lr
    save_json(args.out / "selected_lrs.json", chosen_lrs)
    for seed in args.seeds:
        for arm in args.arms:
            dest = args.out / f"{arm}/seed_{seed}"
            if (dest / "result.json").exists():
                continue
            train(policy, data, arm, seed, chosen_lrs[arm], args.steps, args.batch_size, dest)


if __name__ == "__main__":
    main()
