import json

import numpy as np


def is_numeric_value(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_numeric_values(values):
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return int(round(float(np.mean(values))))
    return round(float(np.mean(values)), 4)


def average_dict_records(records):
    averaged = {}
    for key, first_value in records[0].items():
        values = [record.get(key) for record in records]
        if all(is_numeric_value(value) for value in values):
            averaged[key] = average_numeric_values(values)
        else:
            averaged[key] = first_value
    return averaged


def aggregate_round_results(all_round_results):
    if not all_round_results:
        return []

    expected_len = len(all_round_results[0])
    for run_results in all_round_results:
        if len(run_results) != expected_len:
            raise ValueError("不同随机种子的轮次结果长度不一致，无法直接取平均。")

    aggregated = []
    for idx in range(expected_len):
        records = [run_results[idx] for run_results in all_round_results]
        aggregated.append(average_dict_records(records))
    return aggregated


def save_json(obj, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_metric(round_results, key, ylabel, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [record for record in round_results if key in record]
    if not valid:
        return

    x = [record["round_id"] for record in valid]
    y = [record[key] for record in valid]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Round")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def jain_index(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.sum(values ** 2) <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / (values.size * np.sum(values ** 2)))


def gini_coefficient(values):
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0 or np.sum(values) <= 0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


def compute_fairness_metrics(workers):
    """公平性指标：在至少有一个可用时段的工人上统计被选次数与累计收入的分布。"""
    eligible = [w for w in workers.values() if w.get("active_rounds", 0) > 0]
    selections = [w["selected_rounds"] for w in eligible]
    rewards = [w["cumulative_reward"] for w in eligible]
    return {
        "fairness_jain_selection": round(jain_index(selections), 4),
        "fairness_jain_reward": round(jain_index(rewards), 4),
        "fairness_gini_selection": round(gini_coefficient(selections), 4),
        "fairness_gini_reward": round(gini_coefficient(rewards), 4),
        "fairness_participation_ratio": round(
            sum(1 for s in selections if s > 0) / len(selections), 4
        ) if selections else 0.0,
    }


def compute_detection_metrics(workers):
    """恶意工人识别指标（仅评价使用真实类别）。"""
    if not any("category" in w for w in workers.values()):
        return {}
    true_mal = {wid for wid, w in workers.items() if w["init_category"] == "malicious"}
    pred_mal = {wid for wid, w in workers.items() if w.get("category") == "malicious"}
    tp = len(true_mal & pred_mal)
    precision = tp / len(pred_mal) if pred_mal else 0.0
    recall = tp / len(true_mal) if true_mal else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    true_trusted = {wid for wid, w in workers.items() if w["init_category"] == "trusted"}
    false_mal = len(true_trusted & pred_mal) + len(
        {wid for wid, w in workers.items() if w["init_category"] == "unknown"} & pred_mal
    )
    return {
        "malicious_precision": round(precision, 4),
        "malicious_recall": round(recall, 4),
        "malicious_f1": round(f1, 4),
        "honest_misclassified_as_malicious": false_mal,
    }


def summarize_results(round_results, workers, algorithm, initial_stats=None):
    valid_rounds = [record for record in round_results if record["num_tasks"] > 0]

    def safe_mean_round(key):
        if not valid_rounds:
            return 0.0
        values = [record[key] for record in valid_rounds if key in record]
        return round(float(np.mean(values)), 4) if values else 0.0

    left_workers = [worker for worker in workers.values() if not worker["is_active"]]
    member_workers = [worker for worker in workers.values() if worker.get("is_member", False)]

    def safe_mean_values(values):
        return round(float(np.mean(values)), 4) if values else 0.0

    summary = {
        "algorithm": algorithm.name,
        "selection_logic": algorithm.selection_mode,
        "max_selected_workers_per_round": algorithm.config["K"],
        "total_rounds_recorded": len(round_results),
        "total_non_empty_rounds": len(valid_rounds),
        "avg_num_selected_workers_all_non_empty": safe_mean_round("num_selected_workers"),
        "avg_coverage_rate_all_non_empty": safe_mean_round("coverage_rate"),
        "avg_completion_rate_all_non_empty": safe_mean_round("completion_rate"),
        "avg_avg_quality_all_non_empty": safe_mean_round("avg_quality"),
        "avg_platform_utility_all_non_empty": safe_mean_round("platform_utility"),
        "avg_num_left_workers_this_round_all_non_empty": safe_mean_round("num_left_workers_this_round"),
        "final_cumulative_coverage_rate_all_non_empty": round_results[-1]["cumulative_coverage_rate"] if round_results else 0.0,
        "final_cumulative_completion_rate_all_non_empty": round_results[-1]["cumulative_completion_rate"] if round_results else 0.0,
        "final_cumulative_avg_quality_all_non_empty": round_results[-1]["cumulative_avg_quality"] if round_results else 0.0,
        "final_cumulative_platform_utility": round_results[-1]["cumulative_platform_utility"] if round_results else 0.0,
        "final_num_left_workers": len(left_workers),
        "final_num_member_workers": len(member_workers),
        "final_avg_cumulative_reward": safe_mean_values([worker["cumulative_reward"] for worker in workers.values()]),
        "final_avg_cumulative_cost": safe_mean_values([worker["cumulative_cost"] for worker in workers.values()]),
        "final_avg_selected_rounds": safe_mean_values([worker["selected_rounds"] for worker in workers.values()]),
        "final_avg_active_rounds": safe_mean_values([worker["active_rounds"] for worker in workers.values()]),
    }

    optional_round_keys = [
        "trusted_count",
        "unknown_count",
        "malicious_count",
        "member_count",
        "trusted_member_count",
        "membership_fee_income",
        "bonus_payment",
        "validation_cost",
        "total_platform_cost",
        "net_platform_cost",
        "num_validation_tasks",
        "num_validation_reports",
        "num_reused_validation_reports",
        "num_extra_validation_reports",
        "num_ref_reports",
        "num_unknown_reports",
    ]
    for key in optional_round_keys:
        if valid_rounds and any(key in record for record in valid_rounds):
            summary[f"avg_{key}_all_non_empty"] = safe_mean_round(key)

    if round_results:
        summary["final_cumulative_membership_fee_income"] = round_results[-1].get(
            "cumulative_membership_fee_income", 0.0
        )
        summary["final_cumulative_bonus_payment"] = round_results[-1].get(
            "cumulative_bonus_payment", 0.0
        )
        summary["final_cumulative_validation_cost"] = round_results[-1].get(
            "cumulative_validation_cost", 0.0
        )
        summary["final_cumulative_total_platform_cost"] = round_results[-1].get(
            "cumulative_total_platform_cost", 0.0
        )
        summary["final_cumulative_net_platform_cost"] = round_results[-1].get(
            "cumulative_net_platform_cost", 0.0
        )
        summary["final_cumulative_validation_reports"] = round_results[-1].get(
            "cumulative_validation_reports", 0
        )
        summary["final_cumulative_extra_validation_reports"] = round_results[-1].get(
            "cumulative_extra_validation_reports", 0
        )
        summary["final_cumulative_reused_validation_reports"] = round_results[-1].get(
            "cumulative_reused_validation_reports", 0
        )

    summary.update(compute_fairness_metrics(workers))
    summary.update(compute_detection_metrics(workers))
    if valid_rounds and any("trust_rmse" in record for record in valid_rounds):
        summary["avg_trust_rmse_all_non_empty"] = safe_mean_round("trust_rmse")
        summary["final_trust_rmse"] = round_results[-1].get("trust_rmse", 0.0)
    summary["quality_feedback"] = algorithm.config.get("QUALITY_FEEDBACK", "observable")

    if initial_stats:
        summary.update(initial_stats)

    return summary
