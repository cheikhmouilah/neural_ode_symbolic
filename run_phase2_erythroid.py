"""
Phase 2 — Lancer le pipeline erythroid sur 5 seeds.
À exécuter après que run_real_erythroid.py a été créé.
"""
import subprocess, sys, os, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = os.path.join(ROOT, 'real_data', 'scripts', 'run_real_erythroid.py')

print("Phase 2 — Erythroid dataset, 5 seeds")
print("=" * 50)

for seed in range(5):
    print(f"\n[seed={seed}]", flush=True)
    t = time.time()
    r = subprocess.run(
        [PYTHON, SCRIPT, '--dataset', 'erythroid', '--seed', str(seed)],
        capture_output=False
    )
    print(f"  Exit code: {r.returncode} in {(time.time()-t)/60:.1f} min")

print("\nAll seeds done. Run checks/phase3_audit.py to verify.")
