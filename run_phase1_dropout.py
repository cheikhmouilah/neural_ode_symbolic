"""
Phase 1 — Dropout Benchmark
Lance les variants drop10, drop20, drop40, drop60 sur 5 seeds.
Objectif : montrer que distillation > FD direct quand dropout > 20%.
Pré-enregistré dans PREREGISTRATION.json (amendment 2026-11-22).
"""
import subprocess, sys, os, json, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = os.path.join(ROOT, 'synthetic', 'scripts', 'run_synth.py')

VARIANTS = ['drop10', 'drop20', 'drop40', 'drop60']
SEEDS = [0, 1, 2, 3, 4]

print(f"Phase 1 Dropout Benchmark")
print(f"Variants: {VARIANTS}")
print(f"Seeds: {SEEDS}")
print(f"Total runs: {len(VARIANTS) * len(SEEDS)}")
print("=" * 60)

results_dir = os.path.join(ROOT, 'synthetic', 'results', 'synth')
os.makedirs(results_dir, exist_ok=True)

completed = 0
failed = []
t0 = time.time()

for variant in VARIANTS:
    for seed in SEEDS:
        out_file = os.path.join(results_dir, f'synth_{variant}_seed{seed}.json')
        if os.path.exists(out_file):
            print(f"[SKIP] {variant} seed={seed} (already exists)")
            completed += 1
            continue

        print(f"[RUN ] {variant} seed={seed}...", flush=True)
        t_start = time.time()
        try:
            result = subprocess.run(
                [PYTHON, SCRIPT, '--variant', variant, '--seed', str(seed), '--force'],
                capture_output=True, text=True, timeout=1800  # 30min max
            )
            elapsed = time.time() - t_start
            if result.returncode == 0:
                completed += 1
                print(f"  ✓ Done in {elapsed:.0f}s")
            else:
                print(f"  ✗ FAILED (code {result.returncode})")
                print(f"    stderr: {result.stderr[-300:]}")
                failed.append((variant, seed))
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT after 30min")
            failed.append((variant, seed))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append((variant, seed))

total_time = time.time() - t0
print()
print("=" * 60)
print(f"DONE: {completed}/{len(VARIANTS)*len(SEEDS)} runs in {total_time/60:.1f} min")
if failed:
    print(f"FAILED: {failed}")

# Quick analysis
print()
print("=== PRELIMINARY RESULTS ===")
import numpy as np
import glob

for variant in VARIANTS:
    files = sorted(glob.glob(os.path.join(results_dir, f'synth_{variant}_seed*.json')))
    if not files:
        continue
    sym_f1s, fd_f1s = [], []
    for f in files:
        try:
            r = json.load(open(f))
            sf = r.get('distilled', {}).get('structure', {}).get('F1')
            ff = r.get('direct_fd', {}).get('structure', {}).get('F1')
            if sf is not None: sym_f1s.append(sf)
            if ff is not None: fd_f1s.append(ff)
        except Exception:
            pass
    if sym_f1s and fd_f1s:
        sym_m, fd_m = np.mean(sym_f1s), np.mean(fd_f1s)
        diff = sym_m - fd_m
        verdict = "✓ SYM WINS" if diff > 0.05 else ("≈ TIED" if abs(diff) < 0.05 else "✗ FD WINS")
        print(f"  {variant}: sym={sym_m:.3f}  FD={fd_m:.3f}  diff={diff:+.3f}  {verdict}")
