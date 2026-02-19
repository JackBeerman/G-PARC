#!/usr/bin/env python3
"""
G-PARC Test Suite Runner
========================
Runs all test modules and reports aggregate results.

Usage:
    python tests/run_all.py
"""

import subprocess
import sys
import os
from pathlib import Path

test_dir = Path(__file__).parent

test_files = [
    "test_operators.py",
    "test_shocktube_v2.py",
    "test_river_v2.py",
    "test_elastoplastic_v2.py",
]

print("\n" + "=" * 60)
print("G-PARC FULL TEST SUITE")
print("=" * 60)

total_passed = 0
total_failed = 0
suite_results = []

for test_file in test_files:
    path = test_dir / test_file
    if not path.exists():
        print(f"\n⚠️  {test_file} not found, skipping")
        continue
    
    print(f"\n{'─' * 60}")
    print(f"Running {test_file}...")
    print(f"{'─' * 60}")
    
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=False,
        cwd=str(test_dir.parent),
    )
    
    if result.returncode == 0:
        suite_results.append((test_file, "✅ PASS"))
    else:
        suite_results.append((test_file, "❌ FAIL"))
        total_failed += 1

print(f"\n{'=' * 60}")
print("SUITE SUMMARY")
print(f"{'=' * 60}")
for name, status in suite_results:
    print(f"  {status}  {name}")

n_suites = len(suite_results)
n_passed = n_suites - total_failed
print(f"\n  {n_passed}/{n_suites} test suites passed")
print(f"{'=' * 60}\n")

sys.exit(1 if total_failed > 0 else 0)
