# -*- coding: utf-8 -*-
"""
bridge.py — Python ↔ Julia 桥接层
===================================
设计原则:
  1. 惰性加载: Julia 只在第一次调用热模块时启动 (~5-10 s 预热)
  2. 优雅降级: juliacall 或 Julia 缺失时自动切换到 Python 后备实现
  3. 统一 API:  调用方无需感知后端是 Julia 还是 Python

juliacall 安装: pip install juliacall
Julia   安装:   https://julialang.org/downloads/  (>=1.9)
"""

from __future__ import annotations
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_PKG_ROOT  = Path(__file__).resolve().parent.parent.parent
_JULIA_ENTRY = _PKG_ROOT / "src" / "WeldingHot.jl"
_JULIA_PRJ = _PKG_ROOT               # Project.toml 在仓库根目录


def _prepare_julia_environment() -> None:
    """Expose Julia-bundled tools needed by package precompilation."""
    julia_exe = os.environ.get("PYTHON_JULIAPKG_EXE") or shutil.which("julia")
    if julia_exe:
        resolved = Path(julia_exe).expanduser().resolve()
        candidates = [
            resolved.parent,
            resolved.parent.parent / "libexec" / "julia",
        ]
        for candidate in candidates:
            if (candidate / "lld").is_file():
                path_entries = os.environ.get("PATH", "").split(os.pathsep)
                candidate_s = str(candidate)
                if candidate_s not in path_entries:
                    os.environ["PATH"] = os.pathsep.join([candidate_s, *path_entries])
                break

    for var in ("LD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        value = os.environ.get(var)
        if value and "~" in value:
            os.environ[var] = os.pathsep.join(
                str(Path(part).expanduser()) if part.startswith("~") else part
                for part in value.split(os.pathsep)
            )


# ─────────────────────────────────────────────────────────────────────
class _JuliaBackend:
    """单例 Julia 后端: 管理生命周期与数据转换。"""
    _instance: "_JuliaBackend | None" = None

    def __init__(self):
        self._jl      = None
        self._mod     = None   # WeldingHot module
        self._ready   = False
        self._checked = False

    @classmethod
    def instance(cls) -> "_JuliaBackend":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        """返回 Julia 是否可用 (幂等, 只探测一次)。"""
        if not self._checked:
            self._checked = True
            # import juliacall 不仅可能 ImportError; 当它无法联网定位/下载
            # Julia 运行时, 会抛出 HTTPError 等其他异常。一律视为不可用,
            # 触发纯 Python 后备 (这正是混合架构的优雅降级路径)。
            try:
                import juliacall  # noqa: F401
                self._avail = True
            except BaseException as exc:  # noqa: BLE001
                warnings.warn(f"[bridge] juliacall unavailable ({type(exc).__name__}); "
                              "using Python fallback")
                self._avail = False
        return getattr(self, "_avail", False)

    def load(self) -> bool:
        """启动 Julia 并加载 WeldingHot 模块; 返回是否成功。"""
        if self._ready:
            return True
        if not self.available:
            return False
        try:
            t0 = time.perf_counter()
            print("[bridge] Starting Julia backend (first call ~5-10 s)…",
                  flush=True)
            # 告知 juliacall 使用项目 Project.toml
            os.environ.setdefault("JULIA_PROJECT", str(_JULIA_PRJ))
            _prepare_julia_environment()
            from juliacall import Main as jl
            jl.seval(f'import Pkg; Pkg.activate("{_JULIA_PRJ}"); Pkg.instantiate()')
            jl.seval(f'include("{_JULIA_ENTRY}")')
            self._jl  = jl
            self._mod = jl.WeldingHot
            self._ready = True
            print(f"[bridge] Julia ready in {time.perf_counter()-t0:.1f} s",
                  flush=True)
            return True
        except Exception as exc:
            warnings.warn(f"[bridge] Julia backend failed: {exc} — using Python fallback")
            self._avail = False
            return False

    # ── 数据转换助手 ────────────────────────────────────────────────
    @staticmethod
    def to_np(x: Any) -> np.ndarray:
        """Julia array / scalar → NumPy。"""
        try:
            return np.array(x)
        except Exception:
            return x

    @staticmethod
    def to_jl_vec(arr, jl) -> Any:
        """Python list / NumPy array → Julia Vector{Float64}。"""
        return jl.convert(jl.Vector[jl.Float64], [float(x) for x in arr])

    @property
    def jl(self):
        if not self._ready:
            self.load()
        return self._jl

    @property
    def mod(self):
        if not self._ready:
            self.load()
        return self._mod


# 全局单例
_backend = _JuliaBackend.instance()


def julia_available() -> bool:
    """便捷函数: Julia 后端是否可用。"""
    return _backend.available


def load_julia() -> bool:
    """显式预热 Julia (可选; 否则第一次调用热函数时自动触发)。"""
    return _backend.load()


# ─────────────────────────────────────────────────────────────────────
# 公开 API — 每个函数尝试 Julia, 失败则 Python 后备
# ─────────────────────────────────────────────────────────────────────

def run_goldak_fdm(**kwargs) -> dict:
    """Goldak 3D 瞬态 FDM: Julia (10-50×) 或 Python 后备。"""
    if _backend.load():
        jl  = _backend.jl
        mod = _backend.mod
        kw  = {k: float(v) if isinstance(v, (int, float)) else v
               for k, v in kwargs.items()}
        T, peak, meta = mod.run_goldak_fdm(**kw) if kw else mod.run_goldak_fdm()
        return dict(T=_backend.to_np(T), peak=_backend.to_np(peak),
                    Nx=int(meta.Nx), Ny=int(meta.Ny), Nz=int(meta.Nz),
                    dx=float(meta.dx), T0=float(meta.T0), Tm=float(meta.Tm),
                    backend="julia")
    # Python 后备
    from .thermal import GoldakFDM
    g = GoldakFDM(**{k: v for k, v in kwargs.items()
                    if k in ("Q","eta","v","a","b","cf","cr","ff",
                              "Lx","Ly","Lz","dx","rho","cp","k",
                              "T0","Tm")})
    g.run(t_end=kwargs.get("t_end", 5.0),
          x_start=kwargs.get("x_start", 0.015))
    meta = {"Nx": g.Nx, "Ny": g.Ny, "Nz": g.Nz, "dx": g.dx,
            "T0": g.T0, "Tm": g.Tm}
    return dict(T=g.T, peak=g.peak, **meta, backend="python")


def run_droplet_resonance(fp_arr: np.ndarray,
                          method: str = "vi", **kwargs) -> dict:
    """熔滴共振曲线扫描: Julia 或 Python 后备。"""
    fp_arr = np.asarray(fp_arr, float)
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        p     = mod.DropletParams(**{k: float(v) for k, v in kwargs.items()})
        fp_jl = _backend.to_jl_vec(fp_arr, jl)
        amps  = mod.droplet_resonance_sweep(p, fp_jl, method=method)
        return dict(fp=fp_arr, amp=_backend.to_np(amps), backend="julia")
    from .droplet_vi import DropletOscillatorVI
    d = DropletOscillatorVI(**kwargs)
    amp = d.resonance_sweep(fp_arr, method)
    return dict(fp=fp_arr, amp=amp, backend="python")


def run_droplet_free_energy(t_end: float = None, h: float = None,
                             **kwargs) -> dict:
    """自由振荡能量: 三种积分器对比。"""
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        p = mod.DropletParams(**{k: float(v) for k, v in kwargs.items()})
        T0_ = 1.0 / float(p.f0)
        t_e = t_end or 80 * T0_
        h_  = h or T0_ / 20
        t, E_vi, E_ee, E_ie = mod.droplet_free_energy(p, t_e, h_)
        return dict(t=_backend.to_np(t),
                    E_vi=_backend.to_np(E_vi),
                    E_ee=_backend.to_np(E_ee),
                    E_ie=_backend.to_np(E_ie), backend="julia")
    from .droplet_vi import DropletOscillatorVI
    d = DropletOscillatorVI(**kwargs)
    T0_ = 1.0 / d.f0
    t_e = t_end or 80 * T0_; h_ = h or T0_ / 20
    t, X, V = d.run_vi(0.0, t_e, h_, x0=1e-4)
    E_vi = d.energy(X, V)
    tE, XE = d.run_explicit_euler(0.0, t_e, h_, x0=1e-4)
    tI, XI = d.run_implicit_euler(0.0, t_e, h_, x0=1e-4)
    VE = np.gradient(XE, h_); VI_ = np.gradient(XI, h_)
    return dict(t=t, E_vi=E_vi / E_vi[0],
                E_ee=d.energy(XE, VE) / E_vi[0],
                E_ie=d.energy(XI, VI_) / E_vi[0], backend="python")


def run_robot_passive(**kwargs) -> dict:
    """机器人无驱动能量对比: Julia 或 Python 后备。"""
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        arm = mod.TwoLinkArm(**{k: float(v) for k, v in kwargs.items()
                                  if k in ("m1","m2","l1","l2","g")})
        t_vi, dE_vi, t_rk, dE_rk = mod.passive_compare(arm)
        return dict(t_vi=_backend.to_np(t_vi), dE_vi=_backend.to_np(dE_vi),
                    t_rk=_backend.to_np(t_rk), dE_rk=_backend.to_np(dE_rk),
                    backend="julia")
    from .robot_vi import TwoLinkArm as PyArm
    arm = PyArm(**{k: v for k, v in kwargs.items()
                   if k in ("m1","m2","l1","l2","g")})
    tv, ev, tr, er = arm.passive_compare()
    return dict(t_vi=tv, dE_vi=ev, t_rk=tr, dE_rk=er, backend="python")


def run_robot_seam(**kwargs) -> dict:
    """焊缝跟踪: Julia 或 Python 后备。"""
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        arm = mod.TwoLinkArm()
        t, tip, ref, err = mod.seam_tracking(arm)
        tip = _backend.to_np(tip); ref = _backend.to_np(ref)
        return dict(t=_backend.to_np(t), tip_x=tip[:,0], tip_y=tip[:,1],
                    ref_x=ref[:,0], ref_y=ref[:,1],
                    err=_backend.to_np(err), backend="julia")
    from .robot_vi import TwoLinkArm as PyArm
    arm = PyArm()
    t, tip, ref, err = arm.seam_tracking()
    return dict(t=t, tip_x=tip[:,0], tip_y=tip[:,1],
                ref_x=ref[:,0], ref_y=ref[:,1], err=err, backend="python")


def run_cmt_cycle(**kwargs) -> dict:
    """CMT 触池循环: Julia 或 Python 后备。"""
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        p = mod.ContactParams()
        t, x, x_eq, ph, events = mod.cmt_cycle(p)
        return dict(t=_backend.to_np(t), x=_backend.to_np(x),
                    x_eq=_backend.to_np(x_eq), phase=_backend.to_np(ph),
                    backend="julia")
    from .shortcircuit_vi import ContactCycleVI
    cc = ContactCycleVI()
    out, _ = cc.simulate_cycle(t_end=0.06)
    return dict(t=out[:,0], x=out[:,1], x_eq=out[:,2], phase=out[:,3],
                backend="python")


def run_bounce_comparison(**kwargs) -> dict:
    """弹性反冲能量保真对比: Julia 或 Python 后备。"""
    if _backend.load():
        jl, mod = _backend.jl, _backend.mod
        p = mod.ContactParams()
        Tn, Xn, En = mod.bounce_nonsmooth(p)
        Tp, Xp, Ep = mod.bounce_penalty(p)
        return dict(t_ns=_backend.to_np(Tn), x_ns=_backend.to_np(Xn),
                    E_ns=_backend.to_np(En), t_pen=_backend.to_np(Tp),
                    x_pen=_backend.to_np(Xp), E_pen=_backend.to_np(Ep),
                    backend="julia")
    from .shortcircuit_vi import ContactCycleVI
    cc = ContactCycleVI()
    Tn, Xn, En = cc.bounce_nonsmooth_vi()
    Tp, Xp, Ep = cc.bounce_penalty()
    return dict(t_ns=Tn, x_ns=Xn, E_ns=En/En[0],
                t_pen=Tp, x_pen=Xp, E_pen=Ep/Ep[0], backend="python")
