import csv
import json
import math
from pathlib import Path


SERIES_CONFIG_RMSE = [
    {
        "label": "Random",
        "csv_key": "random",
        "path": "experiment2_random_longrun_round_results.json",
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.8,
        "markersize": 4.2,
        "alpha": 0.95,
        "zorder": 3,
    },
    {
        "label": "UCB-Greedy",
        "csv_key": "cmab",
        "path": "experiment2_cmab_longrun_round_results.json",
        "color": "#54A24B",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.8,
        "markersize": 4.4,
        "alpha": 0.95,
        "zorder": 4,
    },
    {
        "label": "Trust-Aware",
        "csv_key": "cmab_trust",
        "path": "experiment2_cmab_trust_round_results.json",
        "color": "#E45756",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.1,
        "markersize": 4.2,
        "alpha": 0.98,
        "zorder": 4,
    },
    {
        "label": "TruthRide",
        "csv_key": "truthride",
        "path": "experiment2_cmab_trust_pgrd_lgsc_round_results.json",
        "color": "#000000",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 2.9,
        "markersize": 4.8,
        "alpha": 1.0,
        "zorder": 5,
    },
]


WORKER_OPTIONS_FILE = "experiment2_worker_options.json"


def load_workers():
    with Path(WORKER_OPTIONS_FILE).open("r", encoding="utf-8") as f:
        workers = json.load(f)
    return workers


def build_worker_indexes(workers):
    tasks_by_slot = {}
    task_detail_map = {}
    trusted_reference_map = {}

    for worker in workers.values():
        worker_id = int(worker["worker_id"])
        slot_task_map = {}
        detail_map = {}

        for task in worker.get("tasks", []):
            slot_id = int(task["slot_id"])
            task_id = task["task_id"]
            slot_task_map.setdefault(slot_id, []).append(task_id)
            detail_map[task_id] = task

            if worker.get("init_category") == "trusted":
                trusted_reference_map.setdefault((slot_id, task_id), []).append(
                    float(task["task_data"])
                )

        tasks_by_slot[worker_id] = slot_task_map
        task_detail_map[worker_id] = detail_map

    return tasks_by_slot, task_detail_map, trusted_reference_map


def load_round_results(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_task_error_records(round_result, task_detail_map):
    """Offline Fig. 4/5 errors for exactly evaluate_round's covered tasks.

    executed_by is the authoritative normal-sensing worker list. Verification
    labels/references never add, remove, or reweight tasks or reports here.
    Missing/mismatched input is an error, not a reason to silently drop a task.
    """
    slot_id = int(round_result["slot_id"])
    context = f"round {round_result['round_id']}, slot {slot_id}"
    if "task_results" not in round_result:
        raise ValueError(f"{context}: task_results required for task-ground-truth RMSE")

    records = []
    seen_task_ids = set()
    for task in round_result["task_results"]:
        task_id = task["task_id"]
        if task_id in seen_task_ids:
            raise ValueError(f"{context}: duplicate task {task_id!r}")
        seen_task_ids.add(task_id)
        if not task["covered"]:
            continue
        label = f"{context}, task {task_id!r}"
        if int(task["slot_id"]) != slot_id:
            raise ValueError(f"{label}: task-result slot mismatch")
        worker_ids = task.get("executed_by", [])
        if not worker_ids:
            raise ValueError(f"{label}: covered task has no executed_by workers")

        reports = []
        truths = []
        for worker_id in worker_ids:
            detail = task_detail_map.get(int(worker_id), {}).get(task_id)
            if detail is None:
                raise ValueError(f"{label}: missing executed report for worker {worker_id}")
            if detail.get("task_id") != task_id or int(detail["slot_id"]) != slot_id:
                raise ValueError(f"{label}: report task/slot mismatch for worker {worker_id}")
            try:
                report = float(detail["task_data"])
                truth = float(detail["true_value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{label}: invalid task_data/true_value for worker {worker_id}") from exc
            if not math.isfinite(report) or not math.isfinite(truth):
                raise ValueError(f"{label}: non-finite task_data/true_value for worker {worker_id}")
            reports.append(report)
            truths.append(truth)
        if any(truth != truths[0] for truth in truths[1:]):
            raise ValueError(f"{label}: inconsistent true_value across executed workers")

        estimate = math.fsum(reports) / len(reports)
        squared_error = (estimate - truths[0]) ** 2
        if not math.isfinite(squared_error):
            raise ValueError(f"{label}: non-finite squared error")
        records.append({
            "task_id": task_id,
            "executed_by": list(worker_ids),
            "task_estimate": estimate,
            "true_value": truths[0],
            "squared_error": squared_error,
        })
    return records


def _compute_task_rmse_series(round_results, task_detail_map):
    rounds, per_round_rmse, cumulative_rmse = [], [], []
    cumulative_squared_error = 0.0
    cumulative_task_count = 0
    for round_result in round_results:
        records = compute_task_error_records(round_result, task_detail_map)
        squared_error = math.fsum(record["squared_error"] for record in records)
        task_count = len(records)
        cumulative_squared_error += squared_error
        cumulative_task_count += task_count
        # Retain the existing empty-round display convention; no dummy task or
        # zero-error sample is ever added to the cumulative denominator.
        per_round = math.sqrt(squared_error / task_count) if task_count else 0.0
        cumulative = (
            math.sqrt(cumulative_squared_error / cumulative_task_count)
            if cumulative_task_count else 0.0
        )
        rounds.append(int(round_result["round_id"]))
        per_round_rmse.append(round(per_round, 4))
        cumulative_rmse.append(round(cumulative, 4))
    return rounds, per_round_rmse, cumulative_rmse


def compute_rmse_series(
    round_results, tasks_by_slot, task_detail_map, trusted_reference_map,
    *, task_ground_truth=False,
):
    # Only Fig. 4/5 opt in. Other callers (e.g. grid sensitivity plots) retain
    # their existing reference-consistency metric and output unchanged.
    if task_ground_truth:
        return _compute_task_rmse_series(round_results, task_detail_map)

    rounds = []
    per_round_rmse = []
    cumulative_rmse = []
    cumulative_sq_error = 0.0
    cumulative_point_count = 0

    for round_result in round_results:
        slot_id = int(round_result["slot_id"])
        selected_workers = round_result.get("selected_workers", [])
        bid_tasks_map = round_result.get("bid_tasks_map")

        round_sq_errors = []

        for worker_id in selected_workers:
            worker_id = int(worker_id)
            if bid_tasks_map is not None:
                executed_task_ids = bid_tasks_map.get(str(worker_id), [])
            else:
                executed_task_ids = tasks_by_slot.get(worker_id, {}).get(slot_id, [])

            worker_task_map = task_detail_map.get(worker_id, {})
            for task_id in executed_task_ids:
                task_detail = worker_task_map.get(task_id)
                if task_detail is None:
                    continue

                reference_values = trusted_reference_map.get((slot_id, task_id), [])
                if not reference_values:
                    continue

                reference_value = _median(reference_values)
                task_data = float(task_detail["task_data"])
                round_sq_errors.append((task_data - reference_value) ** 2)

        rounds.append(int(round_result["round_id"]))

        if round_sq_errors:
            rmse_t = math.sqrt(sum(round_sq_errors) / len(round_sq_errors))
            cumulative_sq_error += sum(round_sq_errors)
            cumulative_point_count += len(round_sq_errors)
        else:
            rmse_t = 0.0

        per_round_rmse.append(round(rmse_t, 4))

        if cumulative_point_count > 0:
            cumulative_rmse_t = math.sqrt(cumulative_sq_error / cumulative_point_count)
        else:
            cumulative_rmse_t = 0.0
        cumulative_rmse.append(round(cumulative_rmse_t, 4))

    return rounds, per_round_rmse, cumulative_rmse


def save_plot_data_csv(output_csv, rounds, series_rows, series_config):
    headers = ["round"] + [config["csv_key"] for config in series_config]
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, round_id in enumerate(rounds):
            row = [round_id]
            for values in series_rows:
                row.append(values[idx])
            writer.writerow(row)
