# -*- coding: utf-8 -*-
"""冒烟测试: 验证桥接 API 在 Python 后备路径下返回结构正确。
Julia 可用时同样通过 (后端透明)。"""
import numpy as np
import pytest

from welding_hybrid import bridge


def test_julia_probe_is_boolean():
    assert isinstance(bridge.julia_available(), bool)


def test_goldak_fdm_shapes():
    r = bridge.run_goldak_fdm(Q=7755.0, t_end=1.0)
    assert r["T"].shape == (r["Nx"], r["Ny"], r["Nz"])
    assert r["peak"].max() >= r["T0"]
    assert r["backend"] in ("julia", "python")


def test_droplet_resonance_peak_near_f0():
    from welding_hybrid import DropletOscillatorVI
    d = DropletOscillatorVI()
    fp = np.linspace(0.6 * d.f0, 1.4 * d.f0, 21)
    r = bridge.run_droplet_resonance(fp, method="vi")
    peak = fp[np.argmax(r["amp"])]
    assert abs(peak - d.f0) / d.f0 < 0.15      # 共振峰在 f0 附近


def test_vi_beats_implicit_euler_amplitude():
    from welding_hybrid import DropletOscillatorVI
    d = DropletOscillatorVI()
    fp = np.linspace(0.6 * d.f0, 1.4 * d.f0, 21)
    vi = bridge.run_droplet_resonance(fp, method="vi")
    ie = bridge.run_droplet_resonance(fp, method="ie")
    # 隐式 Euler 人工阻尼应显著压低共振峰
    assert ie["amp"].max() < vi["amp"].max()


def test_robot_vi_energy_bounded_vs_rk4_drift():
    r = bridge.run_robot_passive()
    # VI 末段能量误差应远小于 RK4 漂移
    assert r["dE_vi"].max() < r["dE_rk"][-1] * 2


def test_bounce_penalty_injects_energy():
    r = bridge.run_bounce_comparison()
    # 非光滑 VI 能量不增长; 罚函数法虚假注入
    assert r["E_pen"].max() / r["E_pen"][0] > r["E_ns"].max() / r["E_ns"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
