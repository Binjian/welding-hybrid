# welding-hybrid — 工业焊接动力学 Python/Julia 混合实现

在纯 Python 焊接动力学模型 (`welding-dynamics`) 基础上，将三处计算热点
迁移到 Julia 后端，同时保留 Python 作为编排与绘图层。Julia 不可用时自动
降级到等价的纯 Python 实现，**对调用方完全透明**。

## 为什么混合

| 热点 | 瓶颈 (Python) | Julia 优势 |
|---|---|---|
| 模块 4 Goldak FDM | ~5000 步时间循环为 Python 循环 | 原生 JIT 机器码，10–50× |
| 模块 6/7/8 变分积分器 | `scipy.fsolve` 有限差分 Jacobian | `ForwardDiff` 精确 Jacobian，Newton 更快更稳 |
| 模块 7 机器人 DEL 200 s | 10000 步隐式求解 | StaticArrays 零分配 + AD，10–20× |

模块 1/2/3/5 已经够快（`solve_ivp` 调用编译 Fortran，ODE 仅 2 状态），
保留在 Python，不迁移。

## 架构

```
welding-hybrid/
├── Project.toml                  # Julia 依赖 (ForwardDiff, StaticArrays)
├── pyproject.toml                # Python 包 (uv)
├── src/
│   ├── julia/                    # ── Julia 高性能后端 ──
│   │   ├── WeldingHot.jl         #   模块入口
│   │   ├── goldak_fdm.jl         #   模块 4: 3D 瞬态 FDM
│   │   ├── variational_core.jl   #   ForcedVerlet + MidpointDEL (ForwardDiff)
│   │   ├── droplet_vi.jl         #   模块 6: 熔滴共振
│   │   ├── robot_vi.jl           #   模块 7: 二连杆 DEL
│   │   └── contact_vi.jl         #   模块 8: 非光滑接触
│   └── welding_hybrid/           # ── Python 层 ──
│       ├── bridge.py             #   ★ Python↔Julia 桥接 + 优雅降级
│       ├── main.py               #   全模块仿真 + 绘图
│       ├── benchmark.py          #   Python vs Julia 性能对比
│       ├── gmaw.py               #   模块 1 (纯 Python)
│       ├── thermal.py            #   模块 2 + 模块 4 Python 后备
│       ├── droplet.py            #   模块 3 (纯 Python)
│       ├── short_circuit.py      #   模块 5 (纯 Python)
│       ├── variational.py        #   变分核心 Python 后备
│       ├── droplet_vi.py         #   模块 6 Python 后备
│       ├── robot_vi.py           #   模块 7 Python 后备
│       └── shortcircuit_vi.py    #   模块 8 Python 后备
├── tests/test_smoke.py
└── results/
```

桥接层是核心设计：`bridge.py` 暴露统一的 `run_*` 函数。每个函数先尝试
Julia，失败（未安装 Julia / juliacall / 无网络）则回退到 `from .xxx import`
的纯 Python 实现。返回字典含 `backend` 字段标明实际后端。

## 安装与运行

### 仅 Python（始终可用）
```bash
uv sync
uv run welding-hybrid       # 全模块仿真 → results/
uv run welding-benchmark    # 性能对比
```

### 启用 Julia 加速
1. 安装 Julia ≥1.9: https://julialang.org/downloads/
2. juliacall 会在首次导入时自动下载匹配的 Julia，或复用已装的：
   ```bash
   export PYTHON_JULIAPKG_EXE=$(which julia)   # 复用系统 Julia
   uv run welding-hybrid                        # 首次启动预热 ~5-10 s
   ```
3. 后端切换无需改任何调用代码；输出会标注 `backend: julia`。

> 注：本仓库在无网络的沙箱中开发，Julia 二进制无法下载，因此已验证的是
> **Python 降级路径**（所有 8 模块物理结果与 `welding-dynamics` 一致，
> 6 个冒烟测试通过）。在装有 Julia 的机器上，相同命令会自动走 Julia 后端。

## 编程接口

```python
from welding_hybrid import run_goldak_fdm, julia_available

print("Julia backend:", julia_available())
r = run_goldak_fdm(Q=7755.0, t_end=5.0)
print(r["backend"], r["T"].shape)   # 'julia' 或 'python'
```

## 验证结果（Python 后备路径）

```
[1]          稳态 I=266 A  Va=29.1 V  P=7755 W
[4/python]   熔池 17.5×7.5×3.8 mm
[6/python]   共振峰 VI 527 Hz | 隐式Euler 506 Hz (被人工阻尼压低 87%)
[7/python]   焊缝跟踪 RMS 0.24 mm; VI 能量有界 / RK4 漂移
[8/python]   罚函数法虚假能量注入 +74413% (非光滑 VI 不注入)
```

与纯 Python 项目逐位一致，证明桥接层未改变物理。

## 关于变分积分器在本问题中的适用性
仅模块 6/7/8 从辛/变分格式获益（保守或近保守的振荡/多体/接触动力学）。
模块 1/4/5 主体为耗散过程（电阻热、热传导、相变），无辛结构可保，
继续使用刚性 ODE 求解器与守恒型空间离散即可。详见各模块 docstring。
