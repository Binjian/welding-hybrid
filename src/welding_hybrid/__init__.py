# -*- coding: utf-8 -*-
"""welding_hybrid — 工业焊接动力学 Python/Julia 混合实现

热点模块 (Goldak FDM, 变分积分器) 通过 bridge.py 路由到 Julia 后端;
Julia 不可用时自动降级到纯 Python 实现, API 完全一致。
"""
from .bridge import (julia_available, load_julia,
                     run_goldak_fdm, run_droplet_resonance,
                     run_droplet_free_energy, run_robot_passive,
                     run_robot_seam, run_cmt_cycle, run_bounce_comparison)
from .gmaw          import GMAWDynamics
from .thermal       import RosenthalThermal, GoldakFDM
from .droplet       import DropletDynamics
from .short_circuit import ShortCircuitGMAW
from .droplet_vi    import DropletOscillatorVI
from .robot_vi      import TwoLinkArm
from .shortcircuit_vi import ContactCycleVI

__version__ = "1.0.0"
__all__ = ["julia_available", "load_julia",
           "run_goldak_fdm", "run_droplet_resonance",
           "run_droplet_free_energy", "run_robot_passive",
           "run_robot_seam", "run_cmt_cycle", "run_bounce_comparison",
           "GMAWDynamics", "RosenthalThermal", "GoldakFDM",
           "DropletDynamics", "ShortCircuitGMAW",
           "DropletOscillatorVI", "TwoLinkArm", "ContactCycleVI"]
