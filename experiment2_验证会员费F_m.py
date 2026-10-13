import argparse

from experiment2_验证会员费F_m_common import (
    RESULTS_BUNDLE_FILE,
    RUNS_PER_VALUE,
    MEMBERSHIP_FEE_VALUES,
    plot_all_figures,
    run_full_experiment,
)


# ============================================================
# Entry script for membership-fee sensitivity experiment.
# It runs TruthRide under different membership fee F_m values
# and then generates all sensitivity figures and summary statistics.
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", nargs="*", type=int, default=MEMBERSHIP_FEE_VALUES)
    parser.add_argument("--runs", type=int, default=RUNS_PER_VALUE)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    bundle = run_full_experiment(
        membership_fee_values=args.values,
        runs_per_value=args.runs,
        keep_temp=args.keep_temp,
    )

    plot_all_figures(bundle)

    print(f"\nSaved bundle: {RESULTS_BUNDLE_FILE}")


if __name__ == "__main__":
    main()