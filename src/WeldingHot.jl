"""
WeldingHot.jl — 焊接动力学高性能 Julia 后端
=============================================
包含四个子模块:
  goldak_fdm.jl       — Goldak 双椭球 3D 瞬态显式 FDM (热循环/熔池)
  variational_core.jl — ForcedVerlet + MidpointDEL (ForwardDiff 精确 Jacobian)
  droplet_vi.jl       — 熔滴振荡共振曲线
  robot_vi.jl         — 二连杆焊接机器人 DEL 积分
  contact_vi.jl       — 非光滑接触变分模型 (CMT 触池循环)
"""
module WeldingHot

using LinearAlgebra
using ForwardDiff
using StaticArrays

include("julia/goldak_fdm.jl")
include("julia/variational_core.jl")
include("julia/droplet_vi.jl")
include("julia/robot_vi.jl")
include("julia/contact_vi.jl")

end # module WeldingHot
