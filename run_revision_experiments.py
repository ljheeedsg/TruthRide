"""
修订版实验总入口（对应《修改方案》表 2 的 E1–E7）。

输出目录结构：
  results/<dataset>/                 E1 主实验（文件名与原画图脚本一致，可直接在该目录下运行画图脚本）
  results/<dataset>/oracle/          E2 真实质量（oracle）对照
  results/<dataset>/ablation/...     E3 / E6 / E7 / sigma0 敏感性
  results/summary_*.csv              汇总表

Usage:
  python run_pipeline.py                       # 先生成三个数据集的工人与任务数据
  python run_revision_experiments.py           # 运行全部实验
  python run_revision_experiments.py --only E1 E2 --datasets chicago
"""
import argparse
import contextlib
import csv
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ["beijing", "rome", "chicago"]

MAIN_MODES = [
    "random", "epsilon_first", "cmab", "ucb_classic",
    "trust", "beta_rep", "crh_td", "pgrd", "lgsc",
]
ORACLE_MODES = ["cmab", "trust", "lgsc"]

SUMMARY_KEYS = [
    "avg_completion_rate_all_non_empty",
    "avg_avg_quality_all_non_empty",
    "final_cumulative_avg_quality_all_non_empty",
    "final_cumulative_platform_utility",
    "final_num_left_workers",
    "avg_trust_rmse_all_non_empty",
    "malicious_precision",
    "malicious_recall",
    "malicious_f1",
    "final_cumulative_validation_cost",
    "fairness_jain_selection",
    "fairness_jain_reward",
    "fairness_gini_reward",
    "fairness_participation_ratio",
]


def set_dataset(dataset):
    """配置模块在 import 时读取数据集名，这里切换数据集后重新加载相关模块。"""
    import importlib
    os.environ["TRUTHRIDE_DATASET"] = dataset
    import experiment2_重构_配置 as cfg
    importlib.reload(cfg)
    import experiment2_重构_main as main_mod
    importlib.reload(main_mod)
    return main_mod.run_experiment


def run_one(run_experiment, dataset, mode, out_dir, overrides=None, tag=None):
    from experiment2_重构_配置 import MODE_OUTPUTS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "WORKER_OPTIONS_FILE": str(ROOT / "data" / dataset / "experiment2_worker_options.json"),
        "prefix": str(out_dir / MODE_OUTPUTS[mode]["prefix"]),
    }
    params.update(overrides or {})
    with contextlib.redirect_stdout(io.StringIO()):
        _, summary = run_experiment(mode, overrides=params)
    row = {"dataset": dataset, "mode": mode, "setting": tag or "default"}
    row.update({key: summary.get(key, "") for key in SUMMARY_KEYS})
    print(f"  {dataset:8s} {mode:14s} {row['setting']:28s} "
          f"completion={row['avg_completion_rate_all_non_empty']} "
          f"quality={row['avg_avg_quality_all_non_empty']} "
          f"utility={row['final_cumulative_platform_utility']}")
    return row


def write_csv(rows, path):
    if not rows:
        return
    path = ROOT / "results" / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--only", nargs="*", default=["E1", "E2", "E3", "E6", "E7", "SIGMA"])
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    base = {"NUM_EXPERIMENT_RUNS": args.runs}

    tables = {key: [] for key in args.only}
    for dataset in args.datasets:
        run_experiment = set_dataset(dataset)
        res = ROOT / "results" / dataset
        print(f"\n===== {dataset} =====")

        if "E1" in args.only:
            for mode in MAIN_MODES:
                tables["E1"].append(run_one(run_experiment, dataset, mode, res, base))

        if "E2" in args.only:
            for mode in ORACLE_MODES:
                for feedback in ["observable", "oracle"]:
                    out = res if feedback == "observable" else res / "oracle"
                    tables["E2"].append(run_one(
                        run_experiment, dataset, mode, out,
                        {**base, "QUALITY_FEEDBACK": feedback}, tag=f"feedback={feedback}",
                    ))

        if "E3" in args.only:
            for vf in ["linear", "sigmoid", "hard"]:
                tables["E3"].append(run_one(
                    run_experiment, dataset, "lgsc", res / "ablation" / f"selection_value_{vf}",
                    {**base, "SELECTION_VALUE_FUNCTION": vf}, tag=f"selection_value={vf}",
                ))
            for delta in [0.3, 0.4, 0.5, 0.6]:
                tables["E3"].append(run_one(
                    run_experiment, dataset, "lgsc", res / "ablation" / f"delta_{delta}",
                    {**base, "DELTA": delta}, tag=f"delta={delta}",
                ))

        if "E6" in args.only:
            for keep in [1.0, 0.7, 0.5, 0.3]:
                for mode in ["cmab", "trust", "lgsc"]:
                    tables["E6"].append(run_one(
                        run_experiment, dataset, mode, res / "ablation" / f"availability_{keep}",
                        {**base, "AVAILABILITY_KEEP_PROB": keep}, tag=f"availability_keep={keep}",
                    ))

        if "E7" in args.only:
            for frac in [0.25, 0.5, 0.75, 1.0]:
                for mode in ["trust", "beta_rep", "crh_td", "lgsc"]:
                    tables["E7"].append(run_one(
                        run_experiment, dataset, mode, res / "ablation" / f"seed_fraction_{frac}",
                        {**base, "SEED_TRUSTED_FRACTION": frac}, tag=f"seed_trusted_fraction={frac}",
                    ))

        if "SIGMA" in args.only:
            for sigma0 in [0.25, 0.5, 1.0]:
                tables["SIGMA"].append(run_one(
                    run_experiment, dataset, "lgsc", res / "ablation" / f"sigma0_{sigma0}",
                    {**base, "QUALITY_ERROR_SCALE": sigma0}, tag=f"sigma0={sigma0}",
                ))

    names = {
        "E1": "summary_E1_main.csv",
        "E2": "summary_E2_observable_vs_oracle.csv",
        "E3": "summary_E3_threshold_ablation.csv",
        "E6": "summary_E6_opportunistic_mobility.csv",
        "E7": "summary_E7_seed_trusted_fraction.csv",
        "SIGMA": "summary_sigma0_sensitivity.csv",
    }
    for key, rows in tables.items():
        write_csv(rows, names[key])


if __name__ == "__main__":
    main()
