"""
One-command pipeline: generate → features → train → launch dashboard.
Usage: python run_pipeline.py
       python run_pipeline.py --skip-data   (if transactions.csv already exists)
       python run_pipeline.py --skip-train  (if models already trained)
"""

import sys, os, time, subprocess, argparse

def run_step(label: str, script: str):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✅  Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="SPLURGE pipeline")
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip data generation (use existing transactions.csv)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (use existing model files)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Don't launch dashboard after pipeline")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("   💸  SPLURGE — Full Pipeline")
    print("═"*60)

    if not args.skip_data:
        if os.path.exists("data/transactions.csv"):
            size_mb = os.path.getsize("data/transactions.csv") / 1e6
            print(f"\n  ℹ️  Found existing transactions.csv ({size_mb:.1f} MB)")
            ans = input("  Re-generate synthetic data? [y/N]: ").strip().lower()
            if ans == "y":
                run_step("Step 1/3 — Generating synthetic data", "src/generate_data.py")
            else:
                print("  Skipping data generation.")
        else:
            run_step("Step 1/3 — Generating synthetic data", "src/generate_data.py")
    else:
        print("\n  ⏭️  Skipping data generation (--skip-data)")

    run_step("Step 2/3 — Engineering features", "src/features.py")

    if not args.skip_train:
        run_step("Step 3/3 — Training models", "src/model.py")
    else:
        print("\n  ⏭️  Skipping model training (--skip-train)")

    print("\n" + "═"*60)
    print("  ✅  Pipeline complete!")
    print("═"*60)

    if not args.no_dashboard:
        print("\n  🚀  Launching dashboard...")
        print("  Open: http://localhost:8501\n")
        os.system("streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()