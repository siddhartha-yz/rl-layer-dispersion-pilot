"""Read completed independent runs; paired seed/item bootstrap, no cherry picking."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_layer_pilot import ROOT, make_data, save_json

run = ROOT / "runs"
data = make_data()
arms = ["all", "middle", "rotate", "delta_drop"]
names = {"all": "All adapters", "middle": "Fixed middle", "rotate": "Rotate updates", "delta_drop": "Drop RL increments"}
results = {}
preds = {}
for arm in arms:
    paths = sorted((run / arm).glob("seed_*/result.json"))
    if len(paths) != 3:
        raise RuntimeError(f"{arm}: expected 3 completed runs, got {len(paths)}")
    results[arm] = [json.loads(p.read_text()) for p in paths]
    preds[arm] = [np.load(p.parent / "predictions.npz") for p in paths]
baseline = json.loads((run / "baseline.json").read_text())
base_ablation = json.loads((run / "baseline_ablation_subset.json").read_text())
summary = {}
for arm, rr in results.items():
    vals = {"id_accuracy": [r["scores"]["test_id"]["accuracy"] for r in rr],
            "ood_accuracy": [np.mean([r["scores"][s]["accuracy"] for s in ["test_format", "test_large"]]) for r in rr],
            "format_accuracy": [r["scores"]["test_format"]["accuracy"] for r in rr],
            "large_accuracy": [r["scores"]["test_large"]["accuracy"] for r in rr],
            "shuffled_question_accuracy": [r["scores"]["test_question_shuffle"]["accuracy"] for r in rr],
            "max_layer_reward_drop": [r["ablation"]["max_reward_drop"] for r in rr],
            "effective_positive_layers": []}
    # Reject changes smaller than 0.01 percentage point: collapsed policies can
    # otherwise acquire a spurious "effective layer count" from float rounding.
    profiles = np.array([[x["reward_drop"] if x["reward_drop"] > 1e-4 else 0.
                          for x in r["ablation"]["layers"]] for r in rr])
    vals["effective_positive_layers"] = [float(v.sum() ** 2 / (v*v).sum()) if v.sum() else 0. for v in profiles]
    summary[arm] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), seeds=[float(t) for t in v]) for k, v in vals.items()}
    summary[arm]["mechanism_eligible"] = [r["ablation"]["full"]["expected_reward"] > base_ablation["expected_reward"] + .01 for r in rr]
    summary[arm]["middle_sensitivity_share"] = [float(v[9:15].sum()/v.sum()) if v.sum() else 0 for v in profiles]
    summary[arm]["top6_sensitivity_share"] = [float(np.sort(v)[-6:].sum()/v.sum()) if v.sum() else 0 for v in profiles]
    summary[arm]["max_loss_fraction_of_rl_gain"] = [float(r["ablation"]["max_reward_drop"] /
        (r["ablation"]["full"]["expected_reward"] - base_ablation["expected_reward"]))
        if r["ablation"]["full"]["expected_reward"] > base_ablation["expected_reward"] else None for r in rr]


def bootstrap_diff(treatment, control, splits):
    y = np.concatenate([[r["label"] for r in data[s]] for s in splits])
    blocks = np.concatenate([[r["block"] for r in data[s]] for s in splits])
    block_indices = np.array([np.flatnonzero(blocks == b) for b in np.unique(blocks)])
    def correct(arm):
        return np.array([np.concatenate([p[s] for s in splits]).argmax(-1) == y for p in preds[arm]])
    diff = correct(treatment).astype(float) - correct(control)
    rng = np.random.default_rng(451)
    boot = []
    for _ in range(4000):
        si = rng.integers(0, 3, 3)
        ii = block_indices[rng.integers(0, len(block_indices), len(block_indices))].reshape(-1)
        boot.append(diff[si][:, ii].mean())
    return dict(mean=float(diff.mean()), paired_seed_differences=diff.mean(1).tolist(),
                bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                note="Paired resampling of training seeds and 4-question choice blocks. Only 3 seeds; exploratory interval.")


comparisons = {c: bootstrap_diff("delta_drop", c, ["test_format", "test_large"]) for c in ["all", "middle", "rotate"]}
save_json(run / "summary.json", dict(baseline=baseline, arms=summary, delta_drop_ood_differences=comparisons))

lines = ["RL 层间分散：本机小实验", "",
         "模型：Qwen2.5-0.5B-Instruct，24层；冻结原权重，训练各层 rank-8 LoRA。",
         "任务：加减乘四选一，只生成一个答案字母；这属于单步 RL，不是长链推理复现。",
         "四题共享同一组选项、答案各占一项；选项盲猜策略严格为25%。算式跨训练/验证/测试去重。",
         "每组3个训练种子，每次300步，每步16题、每题采样8次；KL系数0.1。",
         "学习率各自用独立调参种子和验证集选择；正式比较使用固定末尾checkpoint。", "",
         "方法\t同分布准确率\t域外准确率\t打乱题目后的准确率"]
for arm in arms:
    m = summary[arm]
    display = {"all": "全部层适配", "middle": "固定中间6层", "rotate": "轮流冻结6层更新", "delta_drop": "随机关闭6层RL增量"}[arm]
    lines.append(f"{display}\t{m['id_accuracy']['mean']*100:.2f}% ± {m['id_accuracy']['std']*100:.2f}\t"
                 f"{m['ood_accuracy']['mean']*100:.2f}% ± {m['ood_accuracy']['std']*100:.2f}\t"
                 f"{m['shuffled_question_accuracy']['mean']*100:.2f}%")
lines += ["", "± 为3个训练种子的样本标准差。域外分数是换表述和更大操作数测试的等权平均。",
          "同分布384题；域外共768题；逐层撤除增量使用32个完整选项组、共128题。", ""]
for control, d in comparisons.items():
    lo, hi = np.array(d["bootstrap95"]) * 100
    lines.append(f"随机关闭增量相对 {names[control]} 的域外变化：{d['mean']*100:+.2f} 个百分点，"
                 f"探索性95%区间 [{lo:+.2f}, {hi:+.2f}]；三个配对种子的变化："
                 + ", ".join(f"{v*100:+.2f}" for v in d["paired_seed_differences"]) + "。")
base_gain = summary['all']['id_accuracy']['mean'] - baseline['test_id']['accuracy']
d = comparisons['all']
promising = (base_gain >= .03 and d['mean'] >= .02 and
             all(v > 0 for v in d['paired_seed_differences']) and
             summary['delta_drop']['id_accuracy']['mean'] >= summary['all']['id_accuracy']['mean'] - .02)
lines.insert(2, "结论：" + ("出现性能初筛信号，机制解释仍待验证。" if promising else
                           "本轮没有验证到稳定的性能提升；随机关闭RL增量有时有效，但跨种子波动和退化明显。") + "\n")
lines += ["", f"普通RL相对预训练模型的同分布提升：{base_gain*100:+.2f} 个百分点。",
          "达到预先约定的性能初筛条件：" + ("是。" if promising else "否。"),
          "逐层撤除敏感性（平均）："]
for arm in arms:
    m = summary[arm]
    lines.append(f"{names[arm]}：撤去最关键层的期望奖励损失 {m['max_layer_reward_drop']['mean']*100:.2f} 个百分点；"
                 f"有效正向敏感层数 {m['effective_positive_layers']['mean']:.2f}；"
                 f"中间6层敏感性份额 {np.mean(m['middle_sensitivity_share'])*100:.1f}%；"
                 f"有足够正向RL增益、适合解释敏感性的运行 {sum(m['mechanism_eligible'])}/3。")
diagnostic = run / "deployment_scale_diagnostic.json"
if diagnostic.exists():
    diag = json.loads(diagnostic.read_text())
    corrected_id = [r["scores"]["test_id"]["accuracy"] for r in diag["runs"]]
    corrected_ood = [np.mean([r["scores"][s]["accuracy"] for s in ["test_format", "test_large"]]) for r in diag["runs"]]
    lines += ["", "补充诊断：检查未做反向缩放的增量dropout是否存在训练/推理强度失配。",
              f"仅用调参种子0的验证集在推理缩放1与0.75之间选择，选中 {diag['selected_scale']}。",
              f"统一应用到三个正式种子后：同分布平均 {np.mean(corrected_id)*100:.2f}%，域外平均 {np.mean(corrected_ood)*100:.2f}%。",
              "三个种子的同分布准确率：" + ", ".join(f"{v*100:.2f}%" for v in corrected_id) + "。",
              "主表保留原先约定的不缩放结果；补充诊断未消除低表现种子。"]
lines += ["", "单层撤除效应不可相加，有效层数只是敏感性代理。忽略低于0.01个百分点的撤除变化，避免浮点误差虚构层数。",
          "没有正向RL增益的运行不适合用撤除敏感性解释学习分散。性能提升本身不能证明分散机制。",
          "置信区间按训练种子和4题选项组配对重采样；仅3个种子，区间用于初筛。",
          "本实验尚未建立论文所报告的单层训练贡献曲线，不能直接判断其1.5B–8B模型上的机制。",
          "协议见 protocol.json，原始数值和预测见 runs/。./reproduce.sh 复核并续跑；./reproduce.sh --rerun 归档旧结果后重新训练。",
          "原论文：https://arxiv.org/abs/2607.01232；模型：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct。"]
(ROOT / "RESULTS.txt").write_text("\n".join(lines) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
x = np.arange(4)
for i, (metric, title) in enumerate([("id_accuracy", "Held-out arithmetic"), ("ood_accuracy", "Format / operand shift")]):
    means = [summary[a][metric]["mean"] * 100 for a in arms]
    stds = [summary[a][metric]["std"] * 100 for a in arms]
    ax[i].bar(x, means, yerr=stds, capsize=4, color=["#718096", "#86a873", "#739cbd", "#cf8c61"])
    for j, a in enumerate(arms):
        ax[i].scatter(np.full(3, j), np.array(summary[a][metric]["seeds"])*100, color="#222", s=15, zorder=3)
    base = baseline["test_id"]["accuracy"] if i == 0 else np.mean([baseline[s]["accuracy"] for s in ["test_format", "test_large"]])
    ax[i].axhline(base * 100, ls="--", color="#555", label="Pretrained")
    ax[i].set_xticks(x, [names[a] for a in arms], rotation=15, ha="right", fontsize=8)
    ax[i].set_ylabel("Greedy accuracy (%)")
    ax[i].set_title(title)
    ax[i].legend(fontsize=8)
fig.suptitle("Qwen2.5-0.5B: one-action RLVR pilot (3 training seeds)")
fig.tight_layout()
fig.savefig(run / "performance.png", dpi=180)

fig, ax = plt.subplots(figsize=(10, 4))
for arm in ["all", "rotate", "delta_drop"]:
    profiles = np.array([[x["reward_drop"] for x in r["ablation"]["layers"]] for r in results[arm]])
    ax.plot(np.arange(24), profiles.mean(0) * 100, marker=".", label=names[arm])
ax.axhline(0, color="#aaa", lw=.8)
ax.set(xlabel="Transformer layer (0-based)", ylabel="Expected reward loss when increment removed (pp)",
       title="Functional sensitivity; single-layer effects are non-additive")
ax.legend()
fig.tight_layout()
fig.savefig(run / "layer_sensitivity.png", dpi=180)
print(json.dumps(dict(arms=summary, comparisons=comparisons), indent=2))
