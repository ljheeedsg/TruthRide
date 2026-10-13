import os
from copy import deepcopy


DATASET_TIME_CONFIG = {
    "beijing": {
        "SLOT_SEC": 600,
        "TOTAL_SLOTS": 86400 // 600,
    },
    "rome": {
        "SLOT_SEC": 375,
        "TOTAL_SLOTS": 144,
        "SOURCE_ROUND_START": 41,
        "SOURCE_ROUND_END": 130,
    },
    "chicago": {
        "SLOT_SEC": 600,
        "TOTAL_SLOTS": 86400 // 600,
    },
}

ACTIVE_DATASET_NAME = os.environ.get("TRUTHRIDE_DATASET", "beijing")


BASE_CONFIG = {
    "WORKER_OPTIONS_FILE": "experiment2_worker_options.json",
    "DATASET_NAME": ACTIVE_DATASET_NAME,
    "SLOT_SEC": DATASET_TIME_CONFIG[ACTIVE_DATASET_NAME]["SLOT_SEC"],
    "TOTAL_SLOTS": DATASET_TIME_CONFIG[ACTIVE_DATASET_NAME]["TOTAL_SLOTS"],
    "PER_ROUND_BUDGET": 1000,
    "K": 10,
    "RANDOM_SEED": 3,
    "NUM_EXPERIMENT_RUNS": 10,
    "SEED_STEP": 1,

    "DELTA": 0.4, # 完成任务的质量阈值，低于该值视为失败
    "UCB_EXPLORATION_ALPHA": 0.5,
    "DEFAULT_INIT_UCB": 0.7,
    "QUALITY_SCORE_LAMBDA": 0.3,
    "UNCERTAINTY_PENALTY_MU": 0.2,
    "ROUND_LEVEL_QUALITY_UPDATE": True,
    "STABILITY_BONUS_GAMMA": 0.12,
    "MAX_NEW_WORKERS_PER_ROUND": 2,
    "UCB_STATE_DELTA_CAP": 0.12,
    "COLD_START_PENALTY_ZERO": 0.18,
    "COLD_START_PENALTY_FEW": 0.08,
    "COLD_START_MIN_OBS": 3,
    "RHO": 10.0,
    "QUALITY_VALUE_R_MAX": 30.0,
    "QUALITY_VALUE_K": 10.0,
    "QUALITY_VALUE_Q0": 0.4,
    "WORKER_COST_RATIO": 0.6,
    "BETA0": -0.5,
    "BETA1": 0.1,
    "BETA2": 0.3,
    "VALIDATION_TOP_M": 3,
    "VALIDATION_REPORT_COST": 2.0,
    "MAX_REF_WORKERS_PER_VALIDATION": 3,
    "MAX_UNKNOWN_WORKERS_PER_VALIDATION": 5,
    "TRUST_INIT_TRUSTED": 1.0,
    "TRUST_INIT_UNKNOWN": 0.5,
    "ETA": 0.1, # 信任更新步长
    "THETA_HIGH": 0.8,
    "THETA_LOW": 0.20,
    "ERROR_GOOD": 0.15, # 误差小于该值视为好
    "ERROR_BAD": 0.35, # 误差大于该值视为坏
    "MEMBERSHIP_FEE": 2, 
    "MEMBER_TASK_RATIO": 0.3, 
    "MEMBER_REWARD_MULTIPLIER": 1.6, # 会员任务奖励乘数
    "NORMAL_REWARD_MULTIPLIER": 1.0, # 非会员任务奖励乘数
    "PGRD_LAMBDA": 1.5, # PGRD损失函数中参考损失的权重
    "PGRD_XI": 4.0, # PGRD中sigmoid函数的参数，控制会员资格概率的分布
    "MEMBERSHIP_THRESHOLD": 0.40, # 会员资格概率阈值
    "SUNK_THRESHOLD": 50,
    "MEMBER_BONUS": 25,
    "RHO_INIT": 1.0, # PGRD初始rho值
    "BETA3": 3.5, # LGSC算法中当前沉没损失对会员资格概率的影响权重
    "BETA4": 4.0, # LGSC算法中累计沉没损失对会员资格概率的影响权重
    # ---------- 修订版新增配置 ----------
    # 质量反馈来源：observable（IEWV，默认）/ oracle（真实质量，仅作上界参考）
    "QUALITY_FEEDBACK": "observable",
    # 可观测质量的误差尺度 sigma0：q~ = 1 - min(1, |x - ref| / sigma0)
    "QUALITY_ERROR_SCALE": 0.5,
    # 平台评价用质量-价值函数：sigmoid（默认）/ linear / hard
    "VALUE_FUNCTION": "sigmoid",
    # 招募目标中的质量价值函数：linear（默认，原方法）/ sigmoid / hard
    "SELECTION_VALUE_FUNCTION": "linear",
    # 平台初始已知的种子可信工人占真实可信工人的比例
    "SEED_TRUSTED_FRACTION": 1.0,
    # 机会式移动模拟：工人每个可用时段的保留概率
    "AVAILABILITY_KEEP_PROB": 1.0,
    "SKIP_EMPTY_ROUNDS": False,
    "GENERATE_LEGACY_PLOTS": False,
}


MODE_OUTPUTS = {
    "random": {
        "prefix": "experiment2_random_longrun",
        "plot_keys": [
            "coverage_rate",
            "completion_rate",
            "avg_quality",
            "cumulative_coverage_rate",
            "cumulative_completion_rate",
            "cumulative_avg_quality",
            "platform_utility",
            "cumulative_platform_utility",
            "num_left_workers_this_round",
            "cumulative_left_workers",
        ],
    },
    "epsilon_first": {
    "prefix": "experiment2_epsilon_first_longrun",
    "plot_keys": [
        "coverage_rate",
        "completion_rate",
        "avg_quality",
        "cumulative_coverage_rate",
        "cumulative_completion_rate",
        "cumulative_avg_quality",
        "platform_utility",
        "cumulative_platform_utility",
        "num_left_workers_this_round",
        "cumulative_left_workers",
    ],
    },
    "cmab": {
        "prefix": "experiment2_cmab_longrun",
        "plot_keys": [
            "coverage_rate",
            "completion_rate",
            "avg_quality",
            "cumulative_coverage_rate",
            "cumulative_completion_rate",
            "cumulative_avg_quality",
            "platform_utility",
            "cumulative_platform_utility",
            "num_left_workers_this_round",
            "cumulative_left_workers",
        ],
    },
    "trust": {
        "prefix": "experiment2_cmab_trust",
        "plot_keys": [
            "coverage_rate",
            "completion_rate",
            "avg_quality",
            "cumulative_coverage_rate",
            "cumulative_completion_rate",
            "cumulative_avg_quality",
            "trusted_count",
            "unknown_count",
            "malicious_count",
            "num_validation_tasks",
            "platform_utility",
            "cumulative_platform_utility",
            "num_left_workers_this_round",
            "cumulative_left_workers",
        ],
    },
    "pgrd": {
        "prefix": "experiment2_cmab_trust_pgrd",
        "plot_keys": [
            "coverage_rate",
            "completion_rate",
            "avg_quality",
            "cumulative_coverage_rate",
            "cumulative_completion_rate",
            "cumulative_avg_quality",
            "trusted_count",
            "unknown_count",
            "malicious_count",
            "num_validation_tasks",
            "platform_utility",
            "cumulative_platform_utility",
            "num_left_workers_this_round",
            "cumulative_left_workers",
            "member_count",
            "trusted_member_count",
            "membership_fee_income",
        ],
    },
    "lgsc": {
        "prefix": "experiment2_cmab_trust_pgrd_lgsc",
        "plot_keys": [
            "coverage_rate",
            "completion_rate",
            "avg_quality",
            "cumulative_coverage_rate",
            "cumulative_completion_rate",
            "cumulative_avg_quality",
            "platform_utility",
            "cumulative_platform_utility",
            "num_left_workers_this_round",
            "cumulative_left_workers",
            "member_count",
            "membership_fee_income",
            "bonus_payment",
        ],
    },
}


_BASELINE_PLOT_KEYS = MODE_OUTPUTS["trust"]["plot_keys"]
MODE_OUTPUTS.update({
    "ucb_classic": {"prefix": "experiment2_ucb_classic_longrun", "plot_keys": MODE_OUTPUTS["cmab"]["plot_keys"]},
    "beta_rep": {"prefix": "experiment2_beta_reputation", "plot_keys": _BASELINE_PLOT_KEYS},
    "crh_td": {"prefix": "experiment2_crh_truth_discovery", "plot_keys": _BASELINE_PLOT_KEYS},
})


def build_config(mode, overrides=None):
    config = deepcopy(BASE_CONFIG)
    dataset_name = config["DATASET_NAME"]
    if dataset_name not in DATASET_TIME_CONFIG:
        raise ValueError(f"不支持的数据集配置: {dataset_name}")
    config.update(DATASET_TIME_CONFIG[dataset_name])
    config["MODE"] = mode
    config.update(MODE_OUTPUTS[mode])

    # if mode == "random":
    #     config["BETA1"] = 0.02
    # elif mode == "cmab":
    #     config["BETA1"] = 0.02
    # elif mode in {"trust", "pgrd", "lgsc"}:
    #     config["BETA1"] = 0.1

    if mode == "epsilon_first":
        config["EPSILON_FIRST_RATIO"] = 0.3
        config["ROUND_LEVEL_QUALITY_UPDATE"] = False

    if mode == "random":
        config["ROUND_LEVEL_QUALITY_UPDATE"] = False

    if overrides:
        config.update(overrides)

    prefix = config["prefix"]
    config["ROUND_RESULTS_FILE"] = f"{prefix}_round_results.json"
    config["SUMMARY_FILE"] = f"{prefix}_summary.json"
    config["ALL_RUNS_FILE"] = f"{prefix}_round_results_all_runs.json"

    config["PLOT_FILES"] = {
        "coverage_rate": f"{prefix}_coverage_rate.png",
        "completion_rate": f"{prefix}_completion_rate.png",
        "avg_quality": f"{prefix}_avg_quality.png",
        "cumulative_coverage_rate": f"{prefix}_cumulative_coverage_rate.png",
        "cumulative_completion_rate": f"{prefix}_cumulative_completion_rate.png",
        "cumulative_avg_quality": f"{prefix}_cumulative_avg_quality.png",
        "platform_utility": f"{prefix}_platform_utility.png",
        "cumulative_platform_utility": f"{prefix}_cumulative_platform_utility.png",
        "num_left_workers_this_round": f"{prefix}_left_workers_per_round.png",
        "cumulative_left_workers": f"{prefix}_left_workers.png",
        "trusted_count": f"{prefix}_trusted_count.png",
        "unknown_count": f"{prefix}_unknown_count.png",
        "malicious_count": f"{prefix}_malicious_count.png",
        "num_validation_tasks": f"{prefix}_validation_count.png",
        "member_count": f"{prefix}_member_count.png",
        "trusted_member_count": f"{prefix}_trusted_member_count.png",
        "membership_fee_income": f"{prefix}_membership_fee_income.png",
        "bonus_payment": f"{prefix}_bonus_payment.png",
    }
    return config
