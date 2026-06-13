# -*- coding: utf-8 -*-
"""
benchmark.py — Python vs Julia 热模块性能对比
============================================
运行: uv run welding-benchmark
输出: results/benchmark_report.txt  +  results/benchmark_speedup.png
"""
import time
import numpy as np
from pathlib import Path

OUT = Path("results"); OUT.mkdir(exist_ok=True)


def _time_fn(fn, n_runs=3):
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
        if i == 0:
            first = result
    return min(times), first


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .bridge import julia_available, load_julia

    lines = ["=" * 58,
             "  welding-hybrid performance benchmark",
             "=" * 58]

    # ── Goldak FDM ────────────────────────────────────────────────
    from .thermal import GoldakFDM
    def py_fdm():
        g = GoldakFDM(Q=7755.0)
        g.run(t_end=5.0); return g

    t_py_fdm, _ = _time_fn(py_fdm, n_runs=2)
    lines.append(f"\n[Module 4  Goldak FDM]")
    lines.append(f"  Python  : {t_py_fdm:.2f} s")

    jl_fdm_time = None
    if julia_available() and load_julia():
        from .bridge import run_goldak_fdm
        def jl_fdm(): return run_goldak_fdm(Q=7755.0, t_end=5.0)
        t_jl_fdm, _ = _time_fn(jl_fdm, n_runs=2)
        jl_fdm_time = t_jl_fdm
        lines.append(f"  Julia   : {t_jl_fdm:.2f} s")
        lines.append(f"  Speedup : {t_py_fdm/t_jl_fdm:.1f}×")
    else:
        lines.append("  Julia   : (unavailable)")

    # ── Droplet resonance sweep ───────────────────────────────────
    from .droplet_vi import DropletOscillatorVI
    d = DropletOscillatorVI()
    fp_arr = np.linspace(0.6*d.f0, 1.4*d.f0, 41)

    def py_res(): return d.resonance_sweep(fp_arr, "vi")
    t_py_res, _ = _time_fn(py_res, n_runs=2)
    lines.append(f"\n[Module 6  Droplet resonance sweep, 41 points]")
    lines.append(f"  Python  : {t_py_res:.2f} s")

    jl_res_time = None
    if julia_available():
        from .bridge import run_droplet_resonance
        def jl_res(): return run_droplet_resonance(fp_arr)
        t_jl_res, _ = _time_fn(jl_res, n_runs=2)
        jl_res_time = t_jl_res
        lines.append(f"  Julia   : {t_jl_res:.2f} s")
        lines.append(f"  Speedup : {t_py_res/t_jl_res:.1f}×")
    else:
        lines.append("  Julia   : (unavailable)")

    # ── Robot passive 200 s ───────────────────────────────────────
    from .robot_vi import TwoLinkArm
    arm = TwoLinkArm()

    def py_rob(): return arm.passive_compare()
    t_py_rob, _ = _time_fn(py_rob, n_runs=1)
    lines.append(f"\n[Module 7  Robot DEL 200 s, h=20 ms]")
    lines.append(f"  Python  : {t_py_rob:.2f} s")

    jl_rob_time = None
    if julia_available():
        from .bridge import run_robot_passive
        def jl_rob(): return run_robot_passive()
        t_jl_rob, _ = _time_fn(jl_rob, n_runs=1)
        jl_rob_time = t_jl_rob
        lines.append(f"  Julia   : {t_jl_rob:.2f} s")
        lines.append(f"  Speedup : {t_py_rob/t_jl_rob:.1f}×")
    else:
        lines.append("  Julia   : (unavailable)")

    lines.append("\n" + "=" * 58)
    report = "\n".join(lines)
    print(report)
    (OUT / "benchmark_report.txt").write_text(report)

    # ── Bar chart ─────────────────────────────────────────────────
    modules = ["Goldak FDM", "Droplet sweep", "Robot DEL 200s"]
    py_times = [t_py_fdm, t_py_res, t_py_rob]
    jl_times = [jl_fdm_time, jl_res_time, jl_rob_time]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(modules))
    bars_py = ax.bar(x - 0.2, py_times, 0.38, label="Python (NumPy/fsolve)")
    bars_jl = ax.bar(x + 0.2,
                     [j if j is not None else 0 for j in jl_times],
                     0.38, label="Julia (JIT / ForwardDiff)")
    for i, (tp, tj) in enumerate(zip(py_times, jl_times)):
        if tj:
            ax.text(i + 0.2, tj + 0.05, f"{tp/tj:.1f}×",
                    ha="center", fontsize=10, color="red")
    ax.set_xticks(x); ax.set_xticklabels(modules)
    ax.set_ylabel("wall time [s]")
    ax.set_title("Python vs Julia: hot-module speedup")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    if not any(jl_times):
        ax.text(0.5, 0.5, "Julia unavailable\n(install Julia + juliacall)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_speedup.png", dpi=140)
    print(f"[benchmark] saved: results/benchmark_report.txt & benchmark_speedup.png")


if __name__ == "__main__":
    run()
