# -*- coding: utf-8 -*-
"""
main.py — welding-hybrid 主程序
  • 调用所有 8 个模块 (通过 bridge.py 自动路由到 Julia 或 Python)
  • 生成对比图到 results/
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
OUT = Path("results"); OUT.mkdir(exist_ok=True)

# ── 导入桥接 API ───────────────────────────────────────────────────
from .bridge import (julia_available,
                     run_goldak_fdm, run_droplet_resonance,
                     run_droplet_free_energy, run_robot_passive,
                     run_robot_seam, run_cmt_cycle, run_bounce_comparison)
# ── 纯 Python 模块 (1, 2, 3, 5 不需要 Julia) ─────────────────────
from .gmaw          import GMAWDynamics
from .thermal       import RosenthalThermal
from .droplet       import DropletDynamics
from .short_circuit import ShortCircuitGMAW


def main():
    backend_tag = "Julia" if julia_available() else "Python (fallback)"
    print(f"[main] hot-module backend: {backend_tag}")

    # ════════════════════════════════════════════════════
    # 模块 1 — GMAW 自调节 (纯 Python)
    # ════════════════════════════════════════════════════
    res = GMAWDynamics().simulate()
    P_ss = float(np.mean(res["P"][-200:]))
    print(f"[1] 稳态 I={res['I'][-1]:.0f} A  Va={res['Va'][-1]:.1f} V  P={P_ss:.0f} W")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    fig.suptitle("Module 1: GMAW self-regulation (Python)")
    for ax, key, lab, c in zip(axes.flat, ["I","la","s","Va"],
                               ["I [A]","arc length [mm]",
                                "stick-out [mm]","V_arc [V]"], "brgm"):
        sc = 1e3 if key in ("la","s") else 1
        ax.plot(res["t"], res[key]*sc, c); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.axvline(0.5, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT/"m1_self_regulation.png", dpi=140)

    # ════════════════════════════════════════════════════
    # 模块 4 — Goldak FDM (Julia 热模块 / Python 后备)
    # ════════════════════════════════════════════════════
    r4 = run_goldak_fdm(Q=P_ss, t_end=5.0)
    T, peak = r4["T"], r4["peak"]
    Nx, Ny, Nz, dx, T0, Tm = (r4[k] for k in ("Nx","Ny","Nz","dx","T0","Tm"))
    x_mm = np.arange(Nx)*dx*1e3; y_mm = np.arange(Ny)*dx*1e3
    melt = T >= Tm
    if melt.any():
        ix, iy, iz = np.where(melt)
        L = (ix.max()-ix.min())*dx*1e3
        W = 2*iy.max()*dx*1e3; D = iz.max()*dx*1e3
        print(f"[4/{r4['backend']}] 熔池 {L:.1f}×{W:.1f}×{D:.1f} mm")

    fig4, a4 = plt.subplots(1, 2, figsize=(11, 4))
    fig4.suptitle(f"Module 4: Goldak FDM ({r4['backend']} backend)")
    lv = [298, 600, 900, 1273, 1773, 2800]
    c0 = a4[0].contourf(x_mm, y_mm, T[:,:,0].T, levels=lv, cmap="hot")
    a4[0].contour(x_mm, y_mm, T[:,:,0].T, levels=[Tm], colors="cyan")
    a4[0].set_title("Top-view T (cyan=melt)"); a4[0].set_xlabel("x [mm]")
    fig4.colorbar(c0, ax=a4[0], label="T [K]")
    c1 = a4[1].contourf(x_mm, -np.arange(Nz)*dx*1e3, peak[:,0,:].T,
                        levels=[298,773,1073,1273,1773,3000], cmap="inferno")
    a4[1].contour(x_mm, -np.arange(Nz)*dx*1e3, peak[:,0,:].T,
                  levels=[1273,1773], colors=["lime","cyan"])
    a4[1].set_title("Peak T: fusion zone & HAZ")
    fig4.colorbar(c1, ax=a4[1]); fig4.tight_layout()
    fig4.savefig(OUT/"m4_goldak_fdm.png", dpi=140)

    # ════════════════════════════════════════════════════
    # 模块 6 — 熔滴共振 (Julia / Python 后备)
    # ════════════════════════════════════════════════════
    from .droplet_vi import DropletOscillatorVI
    d = DropletOscillatorVI()
    fp_arr = np.linspace(0.6*d.f0, 1.4*d.f0, 41)
    r_vi  = run_droplet_resonance(fp_arr, method="vi")
    r_ie  = run_droplet_resonance(fp_arr, method="ie")
    A_an  = d.analytic_fundamental(fp_arr)
    pk_vi = fp_arr[np.argmax(r_vi["amp"])]
    pk_ie = fp_arr[np.argmax(r_ie["amp"])]
    print(f"[6/{r_vi['backend']}] 共振峰: VI {pk_vi:.0f} Hz | "
          f"impl.Euler {pk_ie:.0f} Hz (压低 "
          f"{100*(1-r_ie['amp'].max()/r_vi['amp'].max()):.0f}%)")

    r_free = run_droplet_free_energy()
    t_f, E_vi = r_free["t"], r_free["E_vi"]
    E_ee, E_ie = r_free["E_ee"], r_free["E_ie"]

    fig6, a6 = plt.subplots(1, 2, figsize=(12, 4.2))
    a6[0].semilogy(t_f/t_f[-1], np.maximum(np.abs(E_vi - 1), 1e-12),
                   label="Variational (Verlet)")
    a6[0].semilogy(t_f/t_f[-1], np.maximum(np.abs(E_ee - 1), 1e-12),
                   label="Explicit Euler")
    a6[0].semilogy(t_f/t_f[-1], np.maximum(np.abs(E_ie - 1), 1e-12),
                   label="Implicit Euler")
    a6[0].set_xlabel("t / T_total"); a6[0].set_ylabel("|E/E0 - 1|")
    a6[0].set_title("Free-oscillation energy (h=T0/20)"); a6[0].legend()
    a6[1].plot(fp_arr, A_an*1e3, "k--", label="analytic (fundamental)")
    a6[1].plot(fp_arr, r_vi["amp"]*1e3, "o-", ms=3, label="Variational")
    a6[1].plot(fp_arr, r_ie["amp"]*1e3, "s-", ms=3, label="Implicit Euler")
    a6[1].axvline(d.f0, color="gray", ls=":", lw=0.8)
    a6[1].set_xlabel("pulse freq [Hz]"); a6[1].set_ylabel("amplitude [mm]")
    a6[1].set_title("Pulsed-MIG resonance curve"); a6[1].legend()
    for a in a6: a.grid(alpha=0.3)
    fig6.suptitle(f"Module 6: droplet oscillation ({r_vi['backend']} backend)")
    fig6.tight_layout(); fig6.savefig(OUT/"m6_droplet_vi.png", dpi=140)

    # ════════════════════════════════════════════════════
    # 模块 7 — 机器人 DEL (Julia / Python 后备)
    # ════════════════════════════════════════════════════
    r7p = run_robot_passive()
    r7s = run_robot_seam()
    rms_err = float(np.sqrt(np.mean(r7s["err"]**2))) * 1e3
    print(f"[7/{r7p['backend']}] 能量误差 max_VI={r7p['dE_vi'].max():.2e} "
          f"末段_RK4={r7p['dE_rk'][-1]:.2e} | 跟踪 RMS={rms_err:.2f} mm")

    fig7, a7 = plt.subplots(1, 2, figsize=(12, 4.2))
    a7[0].semilogy(r7p["t_vi"], np.maximum(r7p["dE_vi"], 1e-12),
                   label="Variational (midpoint DEL)")
    a7[0].semilogy(r7p["t_rk"], np.maximum(r7p["dE_rk"], 1e-12),
                   label="RK4 (same h)")
    a7[0].set_xlabel("t [s]"); a7[0].set_ylabel("|E/E0 - 1|")
    a7[0].set_title("Passive energy, 200 s, h=20 ms"); a7[0].legend()
    a7[1].plot(r7s["ref_x"]*1e3, r7s["ref_y"]*1e3, "k--", lw=1.5,
               label="weld seam")
    a7[1].plot(r7s["tip_x"]*1e3, r7s["tip_y"]*1e3, "r", lw=1,
               label=f"torch tip (RMS {rms_err:.2f} mm)")
    a7[1].set_xlabel("x [mm]"); a7[1].set_ylabel("y [mm]")
    a7[1].set_title("Seam tracking (forced DEL)"); a7[1].legend()
    a7[1].axis("equal")
    for a in a7: a.grid(alpha=0.3)
    fig7.suptitle(f"Module 7: welding-robot ({r7p['backend']} backend)")
    fig7.tight_layout(); fig7.savefig(OUT/"m7_robot_vi.png", dpi=140)

    # ════════════════════════════════════════════════════
    # 模块 8 — 非光滑接触 (Julia / Python 后备)
    # ════════════════════════════════════════════════════
    r8c = run_cmt_cycle()
    r8b = run_bounce_comparison()
    inj = (r8b["E_pen"].max() / r8b["E_pen"][0] - 1) * 100
    print(f"[8/{r8c['backend']}] 罚函数法虚假能量注入 +{inj:.0f}%")

    from .shortcircuit_vi import ContactCycleVI
    gap_mm = ContactCycleVI().gap * 1e3

    fig8, a8 = plt.subplots(1, 3, figsize=(14, 4.2))
    a8[0].plot(r8c["t"]*1e3, r8c["x"]*1e3, label="droplet x")
    a8[0].plot(r8c["t"]*1e3, r8c["x_eq"]*1e3, "--", label="wire x_eq")
    a8[0].axhline(gap_mm, color="gray", lw=0.8, label="pool surface")
    a8[0].fill_between(r8c["t"]*1e3, -0.1, gap_mm*1.4,
                       where=r8c["phase"] > 0.5, color="orange", alpha=0.2)
    a8[0].set_xlabel("t [ms]"); a8[0].set_ylabel("x [mm]")
    a8[0].set_title("CMT dip-transfer cycle"); a8[0].legend(fontsize=8)
    a8[1].plot(r8b["t_ns"]*1e3, r8b["x_ns"]*1e3, label="nonsmooth VI")
    a8[1].plot(r8b["t_pen"]*1e3, r8b["x_pen"]*1e3, alpha=0.7,
               label="penalty method")
    a8[1].axhline(0, color="gray", lw=0.8)
    a8[1].set_xlabel("t [ms]"); a8[1].set_title("Bounce trajectories (e=0.85)")
    a8[1].legend(fontsize=8)
    a8[2].plot(r8b["t_ns"]*1e3, r8b["E_ns"], label="nonsmooth VI")
    a8[2].plot(r8b["t_pen"]*1e3, r8b["E_pen"], alpha=0.7,
               label="penalty method")
    a8[2].set_xlabel("t [ms]"); a8[2].set_ylabel("E / E0")
    a8[2].set_title("Energy fidelity"); a8[2].legend(fontsize=8)
    for a in a8: a.grid(alpha=0.3)
    fig8.suptitle(f"Module 8: nonsmooth contact ({r8c['backend']} backend)")
    fig8.tight_layout(); fig8.savefig(OUT/"m8_contact_vi.png", dpi=140)

    print(f"\n✓ All figures saved to results/  (backend: {backend_tag})")


if __name__ == "__main__":
    main()
