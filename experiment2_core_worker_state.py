import math
import random

import numpy as np


def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def get_available_workers(workers, slot_id):
    return [
        worker
        for worker in workers.values()
        if slot_id in worker["available_slots"] and worker["is_active"]
    ]


def _executed_task_ids(worker, slot_id, bid_tasks_map=None):
    if bid_tasks_map is not None:
        return bid_tasks_map.get(worker["worker_id"], [])
    return worker["tasks_by_slot"].get(slot_id, [])


def collect_round_reports(selected_worker_ids, workers, slot_id, bid_tasks_map=None, validation_tasks=None):
    """
    收集平台本轮实际收到的上报数据 x_ij（平台可观测）。
    来源：被招募工人执行任务的上报 + 信任验证阶段额外收集的上报。
    返回 {task_id: {worker_id: x_ij}}。
    """
    reports = {}
    for worker_id in selected_worker_ids:
        worker = workers[worker_id]
        for task_id in _executed_task_ids(worker, slot_id, bid_tasks_map):
            if task_id in worker["task_map"]:
                reports.setdefault(task_id, {})[worker_id] = float(worker["task_map"][task_id]["task_data"])

    for item in validation_tasks or []:
        task_id = item.get("task_id")
        for worker_id in list(item.get("ref_worker_ids", [])) + list(item.get("unknown_worker_ids", [])):
            worker = workers.get(worker_id)
            if worker is None or task_id not in worker["task_map"]:
                continue
            reports.setdefault(task_id, {})[worker_id] = float(worker["task_map"][task_id]["task_data"])
    return reports


def observable_task_quality(worker_id, task_id, round_reports, workers, config):
    """
    基于上报一致性的可观测质量（不访问 q_ij 与 v_j）：
        ref = 其他可信工人上报的中位数；若无可信工人，则取其他非恶意工人上报的中位数（留一法）
        q~_ij = 1 - min(1, |x_ij - ref| / sigma0)
    若该任务没有其他上报，则返回 None（本轮不产生质量观测）。
    """
    task_reports = round_reports.get(task_id, {})
    if worker_id not in task_reports:
        return None

    others = {wid: value for wid, value in task_reports.items() if wid != worker_id}
    if not others:
        return None

    trusted_values = [
        value for wid, value in others.items()
        if workers[wid].get("category") == "trusted"
    ]
    if trusted_values:
        reference = float(np.median(trusted_values))
    else:
        peer_values = [
            value for wid, value in others.items()
            if workers[wid].get("category") != "malicious"
        ]
        if not peer_values:
            return None
        reference = float(np.median(peer_values))

    sigma0 = max(float(config.get("QUALITY_ERROR_SCALE", 0.5)), 1e-8)
    error = abs(task_reports[worker_id] - reference)
    return 1.0 - min(1.0, error / sigma0)


def update_worker_statistics(
    selected_worker_ids,
    workers,
    slot_id,
    config,
    bid_tasks_map=None,
    round_reports=None,
):
    """
    更新工人质量估计 (n_obs, avg_quality)。

    QUALITY_FEEDBACK = "observable"（默认，符合 IEWV）：仅使用上报数据之间的一致性；
    QUALITY_FEEDBACK = "oracle"：使用仿真生成的真实质量 q_ij，仅作为性能上界参考。
    """
    total_new_observations = 0
    round_level_quality_update = bool(config.get("ROUND_LEVEL_QUALITY_UPDATE", False))
    use_oracle = str(config.get("QUALITY_FEEDBACK", "observable")).lower() == "oracle"
    round_reports = round_reports or {}

    for worker_id in selected_worker_ids:
        worker = workers[worker_id]
        task_ids = _executed_task_ids(worker, slot_id, bid_tasks_map)
        qualities = []

        for task_id in task_ids:
            if task_id not in worker["task_map"]:
                continue
            if use_oracle:
                qualities.append(float(worker["task_map"][task_id]["quality"]))
            else:
                q_obs = observable_task_quality(worker_id, task_id, round_reports, workers, config)
                if q_obs is not None:
                    qualities.append(q_obs)

        if not qualities:
            continue

        old_n = worker["n_obs"]
        old_avg = worker["avg_quality"]

        if round_level_quality_update:
            round_avg_quality = sum(qualities) / len(qualities)
            new_n = old_n + 1
            new_avg = ((old_n * old_avg) + round_avg_quality) / new_n
            total_new_observations += 1
        else:
            new_obs = len(qualities)
            new_total_quality = old_n * old_avg + sum(qualities)
            new_n = old_n + new_obs
            new_avg = new_total_quality / new_n
            total_new_observations += new_obs

        worker["n_obs"] = new_n
        worker["avg_quality"] = float(new_avg)

    return total_new_observations


def update_worker_reward_cost(
    selected_worker_ids,
    workers,
    worker_cost_ratio,
    mode="base",
    config=None,
    bid_tasks_map=None,
    member_task_ids=None,
):
    """
    返回平台本轮实际支付给被招募工人的基础报酬总额（不含 LGSC 奖励），
    供平台效用计算使用，保证平台支出与工人收入口径一致。
    感知成本只与工人报价相关，不随会员报酬倍率变化。
    """
    selected_set = set(selected_worker_ids)
    member_task_ids = set(member_task_ids or [])
    config = config or {}
    total_payment = 0.0

    for worker_id, worker in workers.items():
        if worker_id in selected_set:
            reward_multiplier = 1.0
            if mode in {"pgrd", "lgsc"} and worker.get("is_member"):
                task_ids = (
                    bid_tasks_map.get(worker_id, [])
                    if bid_tasks_map is not None else []
                )
                if any(task_id in member_task_ids for task_id in task_ids):
                    reward_multiplier = float(config.get("MEMBER_REWARD_MULTIPLIER", 1.0))
                else:
                    reward_multiplier = float(config.get("NORMAL_REWARD_MULTIPLIER", 1.0))

            reward_t = reward_multiplier * float(worker["bid_price"])
            cost_t = worker_cost_ratio * float(worker["bid_price"])
            total_payment += reward_t
            worker["recent_reward"] = reward_t
            worker["cumulative_reward"] += reward_t
            worker["cumulative_cost"] += cost_t
            worker["selected_rounds"] += 1
        else:
            worker["recent_reward"] = 0.0

    return round(total_payment, 4)


def update_worker_bonus_rewards(workers, bonus_reward_map):
    for worker_id, bonus_reward in bonus_reward_map.items():
        if worker_id not in workers:
            continue
        worker = workers[worker_id]
        reward_bonus = float(bonus_reward)
        worker["recent_reward"] += reward_bonus
        worker["cumulative_reward"] += reward_bonus


def update_worker_leave_state(workers, round_id, selected_worker_ids, config, mode):
    left_worker_ids = []
    leave_probabilities = []
    selected_set = set(selected_worker_ids)

    for worker_id, worker in workers.items():
        if not worker["is_active"]:
            continue

        if worker_id not in selected_set:
            worker["leave_probability"] = 0.0
            continue

        avg_reward = worker["cumulative_reward"] / max(1, worker["selected_rounds"])

        if mode == "lgsc":
            member_flag = 1 if worker.get("is_member", False) else 0
            sunk_threshold = max(float(config.get("SUNK_THRESHOLD", 1.0)), 1e-8)
            sunk_progress = float(worker.get("sunk_value", 0.0)) / sunk_threshold
            leave_probability = sigmoid(
                float(config["BETA0"])
                + float(config["BETA1"]) * float(worker["cumulative_cost"])
                - float(config["BETA2"]) * float(avg_reward)
                - float(config.get("BETA3", 0.0)) * member_flag
                - float(config.get("BETA4", 0.0)) * sunk_progress
            )
        else:
            leave_probability = sigmoid(
                float(config["BETA0"])
                + float(config["BETA1"]) * float(worker["cumulative_cost"])
                - float(config["BETA2"]) * float(avg_reward)
            )

        worker["leave_probability"] = float(leave_probability)
        leave_probabilities.append(float(leave_probability))

        if random.random() < leave_probability:
            worker["is_active"] = False
            worker["left_round_id"] = round_id
            if "is_member" in worker:
                worker["is_member"] = False
            left_worker_ids.append(worker["worker_id"])

    avg_leave_probability = float(np.mean(leave_probabilities)) if leave_probabilities else 0.0
    return {
        "left_worker_ids": left_worker_ids,
        "num_left_workers_this_round": len(left_worker_ids),
        "avg_leave_probability": round(avg_leave_probability, 4),
    }


def update_active_rounds(available_workers):
    for worker in available_workers:
        worker["active_rounds"] += 1
