"""
外部对比基线（回应审稿人 1 关于 state-of-the-art 基线的意见）。

1. ClassicUCBAlgorithm  —— 经典 CMAB-UCB 质量感知招募：
   按 (UCB 质量边际增益 / 报价) 贪心选人，不含本文的冷启动惩罚、稳定性奖励、
   不确定性惩罚、新工人数量限制与状态截断项。
2. BetaReputationAlgorithm —— Beta 信誉模型（Jøsang & Ismail, 2002）：
   沿用本文的验证任务与上报收集方式，但信任值由 Beta 分布后验均值
   (alpha + 1) / (alpha + beta + 2) 给出。
3. CRHTruthDiscoveryAlgorithm —— CRH 真值发现（Li et al., SIGMOD 2014）：
   不依赖可信工人参考值，对验证任务上的全部上报做加权聚合，
   并按累计加权误差迭代估计工人可靠度 w_i = log(sum_k D_k / D_i)。

三个基线的招募模块都只使用可观测信息，与 TruthRide 的 IEWV 设定一致。
"""
import math

import numpy as np

from experiment2_重构_CMAB算法 import CMABAlgorithm
from experiment2_重构_Trust算法 import (
    TrustCMABAlgorithm,
    _collect_validation_worker_ids,
    generate_validation_tasks_by_grid,
)


CLASSIC_UCB_OVERRIDES = {
    "QUALITY_SCORE_LAMBDA": 0.0,
    "UNCERTAINTY_PENALTY_MU": 0.0,
    "STABILITY_BONUS_GAMMA": 0.0,
    "COLD_START_PENALTY_ZERO": 0.0,
    "COLD_START_PENALTY_FEW": 0.0,
    "UCB_STATE_DELTA_CAP": 1.0,
}


class ClassicUCBAlgorithm(CMABAlgorithm):
    def __init__(self, config):
        config = dict(config)
        config.update(CLASSIC_UCB_OVERRIDES)
        config["MAX_NEW_WORKERS_PER_ROUND"] = int(config["K"])
        super().__init__(config)
        self.name = "ucb_classic"
        self.loader_mode = "base"
        self.selection_mode = "classic_cmab_ucb_marginal_gain_over_bid"


def _relative_error(value, reference):
    if abs(reference) < 1e-12:
        return abs(value - reference)
    return abs(value - reference) / abs(reference)


def _assign_category(worker, trust, config):
    worker["trust"] = float(trust)
    if trust >= float(config["THETA_HIGH"]):
        worker["category"] = "trusted"
    elif trust <= float(config["THETA_LOW"]):
        worker["category"] = "malicious"
        worker["is_member"] = False
    else:
        worker["category"] = "unknown"


def _collect_reports(item, available_workers, workers, slot_id, config):
    task_id = item["task_id"]
    executed = item.get("executed_by", [])
    reused_ref, extra_ref = _collect_validation_worker_ids(
        task_id, executed, available_workers, workers, slot_id,
        int(config.get("MAX_REF_WORKERS_PER_VALIDATION", 3)), "trusted",
    )
    reused_unk, extra_unk = _collect_validation_worker_ids(
        task_id, executed, available_workers, workers, slot_id,
        int(config.get("MAX_UNKNOWN_WORKERS_PER_VALIDATION", 5)), "unknown",
    )
    return reused_ref + extra_ref, reused_unk + extra_unk, len(extra_ref) + len(extra_unk), len(reused_ref) + len(reused_unk)


class _ValidationBaseline(TrustCMABAlgorithm):
    """复用 TruthRide 的验证任务生成与上报收集，仅替换信任更新规则。"""

    def _update_trust(self, task_id, trusted_ids, unknown_ids, workers):
        raise NotImplementedError

    def update(self, feedback):
        workers = feedback["workers"]
        available_workers = feedback["available_workers"]
        slot_id = feedback["slot_id"]
        validation_tasks, validation_candidates = generate_validation_tasks_by_grid(
            available_workers=available_workers,
            workers=workers,
            task_grid_map=feedback["task_grid_map"],
            eval_result=feedback["eval_result"],
            slot_id=slot_id,
            top_m=self.config["VALIDATION_TOP_M"],
        )

        records = []
        trust_update_records = []
        total_extra = total_reused = total_ref = total_unknown = 0
        for item in validation_tasks:
            trusted_ids, unknown_ids, n_extra, n_reused = _collect_reports(
                item, available_workers, workers, slot_id, self.config
            )
            if not unknown_ids or (self.requires_reference and not trusted_ids):
                continue
            if len(trusted_ids) + len(unknown_ids) < 2:
                continue
            total_extra += n_extra
            total_reused += n_reused
            total_ref += len(trusted_ids)
            total_unknown += len(unknown_ids)
            records.append({
                "task_id": item["task_id"],
                "grid_id": item.get("grid_id"),
                "ref_worker_ids": trusted_ids,
                "unknown_worker_ids": unknown_ids,
                "skipped": False,
            })
            trust_update_records.extend(
                self._update_trust(item["task_id"], trusted_ids, unknown_ids, workers)
            )

        trusted_count, unknown_count, malicious_count = self._count_categories(workers)
        return {
            "num_validation_tasks": len(validation_tasks),
            "validation_tasks": records,
            "validation_candidates": validation_candidates,
            "trust_update_records": trust_update_records,
            "validation_cost": round(float(self.config.get("VALIDATION_REPORT_COST", 0.0)) * total_extra, 4),
            "num_validation_reports": total_ref + total_unknown,
            "num_reused_validation_reports": total_reused,
            "num_extra_validation_reports": total_extra,
            "num_ref_reports": total_ref,
            "num_unknown_reports": total_unknown,
            "trusted_count": trusted_count,
            "unknown_count": unknown_count,
            "malicious_count": malicious_count,
        }


class BetaReputationAlgorithm(_ValidationBaseline):
    requires_reference = True

    def __init__(self, config):
        super().__init__(config)
        self.name = "beta_rep"
        self.selection_mode = "cmab_plus_beta_reputation"

    def _update_trust(self, task_id, trusted_ids, unknown_ids, workers):
        base_v = float(np.median([workers[w]["task_map"][task_id]["task_data"] for w in trusted_ids]))
        out = []
        for worker_id in unknown_ids:
            worker = workers[worker_id]
            error = _relative_error(float(worker["task_map"][task_id]["task_data"]), base_v)
            worker.setdefault("beta_alpha", 0.0)
            worker.setdefault("beta_beta", 0.0)
            if error <= float(self.config["ERROR_GOOD"]):
                worker["beta_alpha"] += 1.0
            elif error > float(self.config["ERROR_BAD"]):
                worker["beta_beta"] += 1.0
            trust = (worker["beta_alpha"] + 1.0) / (worker["beta_alpha"] + worker["beta_beta"] + 2.0)
            _assign_category(worker, trust, self.config)
            out.append({"task_id": task_id, "worker_id": worker_id, "error": round(error, 4), "new_trust": round(trust, 4)})
        return out


class CRHTruthDiscoveryAlgorithm(_ValidationBaseline):
    requires_reference = False

    def __init__(self, config):
        super().__init__(config)
        self.name = "CRH-inspired Online"
        self.selection_mode = "cmab_plus_crh_inspired_online"
        self.min_obs = int(config.get("CRH_MIN_OBS", 3))

    def _reliability_weights(self, worker_ids, workers):
        """Use only residuals recorded before the current verification event."""
        worker_ids = list(worker_ids)
        if not worker_ids:
            return np.array([], dtype=float), []

        historical_mse = []
        for worker_id in worker_ids:
            worker = workers[worker_id]
            residual_history = worker.get("crh_residual_history")
            if residual_history:
                valid_residuals = [
                    float(value)
                    for value in residual_history
                    if np.isfinite(float(value)) and float(value) >= 0.0
                ]
                mse = float(np.mean(valid_residuals)) if valid_residuals else None
            else:
                # Backward-compatible fallback for a worker state carrying only
                # the old aggregate history fields.
                obs = int(worker.get("crh_obs", 0))
                dist = float(worker.get("crh_dist", 0.0))
                mse = max(0.0, dist / obs) if obs > 0 and np.isfinite(dist) else None
            historical_mse.append(mse)

        known_mse = [value for value in historical_mse if value is not None]
        if not known_mse:
            return np.full(len(worker_ids), 1.0 / len(worker_ids)), historical_mse

        # A worker without prior observations receives a neutral prior equal to
        # the cohort median; current-event residuals are not used for imputation.
        neutral_prior = float(np.median(known_mse))
        errors = np.array(
            [neutral_prior if value is None else value for value in historical_mse],
            dtype=float,
        )
        # Keep the legacy CRH log-inverse-error form, with its 1e-3 stabilizer.
        stabilized_errors = np.maximum(errors, 0.0) + 1e-3
        raw_weights = np.log(stabilized_errors.sum() / stabilized_errors) + 1e-6
        raw_weights = np.nan_to_num(raw_weights, nan=1.0, posinf=1.0, neginf=1e-6)
        raw_weights = np.maximum(raw_weights, 1e-6)
        weights = raw_weights / raw_weights.sum()
        return weights, historical_mse

    def _update_trust(self, task_id, trusted_ids, unknown_ids, workers):
        ids = list(dict.fromkeys(trusted_ids + unknown_ids))
        if not ids:
            return []

        values = np.array([float(workers[w]["task_map"][task_id]["task_data"]) for w in ids])
        for w in ids:
            workers[w].setdefault("crh_residual_history", [])
            workers[w].setdefault("crh_dist", 0.0)
            workers[w].setdefault("crh_obs", 0)

        # Strict online order: past history -> current weights -> one estimate.
        weights, prior_mse = self._reliability_weights(ids, workers)
        truth = float(np.dot(weights, values))
        residuals = values - truth

        # Append current-event residuals only after the estimate is fixed; they
        # can affect reliability only at a later verification event.
        for w_id, residual in zip(ids, residuals):
            residual_sq = float(residual ** 2)
            workers[w_id]["crh_residual_history"].append(residual_sq)
            workers[w_id]["crh_dist"] += residual_sq
            workers[w_id]["crh_obs"] += 1

        # 可靠度归一化：以所有已观测工人的平均误差为尺度
        observed = [w for w in workers.values() if w.get("crh_obs", 0) > 0]
        mean_dist = np.mean([w["crh_dist"] / w["crh_obs"] for w in observed]) + 1e-6

        out = []
        for worker_id in unknown_ids:
            worker = workers[worker_id]
            if worker["crh_obs"] < self.min_obs:
                continue
            avg_dist = worker["crh_dist"] / worker["crh_obs"]
            trust = 1.0 / (1.0 + math.exp(4.0 * (avg_dist / mean_dist - 1.0)))
            _assign_category(worker, trust, self.config)
            worker_index = ids.index(worker_id)
            out.append({
                "task_id": task_id,
                "worker_id": worker_id,
                "truth": round(truth, 4),
                "new_trust": round(trust, 4),
                "reliability_weight": round(float(weights[worker_index]), 8),
                "prior_residual_mse": (
                    None if prior_mse[worker_index] is None
                    else round(float(prior_mse[worker_index]), 8)
                ),
                "current_residual_sq": round(float(residuals[worker_index] ** 2), 8),
            })
        return out
