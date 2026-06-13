# robot_vi.jl — 焊接机器人二连杆 MidpointDEL + ForwardDiff
# StaticArrays 保证小矩阵运算零分配; ForwardDiff 精确 Jacobian
# 消除 Python fsolve 有限差分噪声与收敛警告

struct TwoLinkArm
    m1::Float64; m2::Float64
    l1::Float64; l2::Float64
    g ::Float64
end
TwoLinkArm(; m1=4.0, m2=2.5, l1=0.40, l2=0.35, g=9.81) =
    TwoLinkArm(m1, m2, l1, l2, g)

# ── 拉格朗日量 ────────────────────────────────────────────────────
function lagrangian(arm::TwoLinkArm, q::AbstractVector, qd::AbstractVector)
    q1,q2,w1,w2 = q[1],q[2],qd[1],qd[2]
    l1,l2,m1,m2,g = arm.l1,arm.l2,arm.m1,arm.m2,arm.g
    v1sq = (l1*w1)^2
    v2sq = (l1*w1)^2 + (l2*(w1+w2))^2 + 2l1*l2*w1*(w1+w2)*cos(q2)
    T    = 0.5m1*v1sq + 0.5m2*v2sq
    V    = m1*g*l1*sin(q1) + m2*g*(l1*sin(q1) + l2*sin(q1+q2))
    return T - V
end

function total_energy(arm::TwoLinkArm, q::AbstractVector, qd::AbstractVector)
    q1,q2,w1,w2 = q[1],q[2],qd[1],qd[2]
    l1,l2,m1,m2,g = arm.l1,arm.l2,arm.m1,arm.m2,arm.g
    v1sq = (l1*w1)^2
    v2sq = (l1*w1)^2 + (l2*(w1+w2))^2 + 2l1*l2*w1*(w1+w2)*cos(q2)
    T_kin = 0.5m1*v1sq + 0.5m2*v2sq
    V_pot = m1*g*l1*sin(q1) + m2*g*(l1*sin(q1) + l2*sin(q1+q2))
    return T_kin + V_pot
end

# ── 动力学 (RK4 基线与启动用) ────────────────────────────────────
function arm_accel(arm::TwoLinkArm, q::AbstractVector, qd::AbstractVector,
                   tau=nothing)
    q1,q2,w1,w2 = q[1],q[2],qd[1],qd[2]
    l1,l2,m1,m2,g = arm.l1,arm.l2,arm.m1,arm.m2,arm.g
    c2 = cos(q2);  s2 = sin(q2)
    M = SA[(m1+m2)*l1^2 + m2*l2^2 + 2m2*l1*l2*c2   m2*l2^2+m2*l1*l2*c2;
           m2*l2^2+m2*l1*l2*c2                       m2*l2^2]
    hh = m2*l1*l2*s2
    C  = SA[-hh*(2w1*w2+w2^2), hh*w1^2]
    G  = SA[(m1+m2)*g*l1*cos(q1)+m2*g*l2*cos(q1+q2), m2*g*l2*cos(q1+q2)]
    rhs = (tau === nothing ? SA[0.0,0.0] : tau) .- C .- G
    return M \ rhs
end

# ── 正运动学 / 逆运动学 ──────────────────────────────────────────
function fk_tip(arm::TwoLinkArm, q::AbstractVector)
    SA[arm.l1*cos(q[1]) + arm.l2*cos(q[1]+q[2]),
       arm.l1*sin(q[1]) + arm.l2*sin(q[1]+q[2])]
end

function ik(arm::TwoLinkArm, x, y)
    c2 = clamp((x^2+y^2-arm.l1^2-arm.l2^2)/(2arm.l1*arm.l2), -1.0, 1.0)
    q2 = -acos(c2)
    q1 = atan(y, x) - atan(arm.l2*sin(q2), arm.l1+arm.l2*cos(q2))
    SA[q1, q2]
end

gravity_comp(arm::TwoLinkArm, q) = begin
    q1,q2 = q[1],q[2]
    l1,l2,m1,m2,g = arm.l1,arm.l2,arm.m1,arm.m2,arm.g
    SA[(m1+m2)*g*l1*cos(q1)+m2*g*l2*cos(q1+q2), m2*g*l2*cos(q1+q2)]
end

# ── 演示 A: 无驱动能量保持对比 ───────────────────────────────────
"""
    passive_compare(arm; q0, t_end, h) -> (t_vi, ΔE_vi, t_rk, ΔE_rk)

DEL 变分积分器 vs RK4 (相同步长) 能量误差绝对值.
VI 误差有界振荡; RK4 单调漂移.
"""
function passive_compare(arm::TwoLinkArm;
        q0=SA[1.05, 0.30], t_end=200.0, h=0.02)
    v0 = SA[0.0, 0.0]
    L  = (q, qd) -> lagrangian(arm, q, qd)
    ac = (q, v, t)  -> arm_accel(arm, q, v)

    # VI
    t_vi, Q_vi = del_run(L, q0, v0, t_end, h; acc=ac)
    Vc = [(Q_vi[k+2,:] .- Q_vi[k,:])./(2h) for k in 1:size(Q_vi,1)-2]
    E0 = total_energy(arm, q0, v0)
    dE_vi = [abs(total_energy(arm, Q_vi[k+1,:], Vc[k])/E0 - 1.0)
             for k in eachindex(Vc)]

    # RK4
    t_rk, Q_rk, V_rk = rk4_run(ac, q0, v0, t_end, h)
    dE_rk = [abs(total_energy(arm, Q_rk[k,:], V_rk[k,:])/E0 - 1.0)
             for k in 1:size(Q_rk,1)]

    return t_vi[2:end-1], dE_vi, t_rk, dE_rk
end

# ── 演示 B: 焊缝跟踪 (强迫 DEL) ─────────────────────────────────
"""
    seam_tracking(arm; p_start, p_end, t_weld, h, Kp, Kd)
               -> (t, tip_xy, ref_xy, err)
"""
function seam_tracking(arm::TwoLinkArm;
        p_start=SA[0.30,0.10], p_end=SA[0.60,0.25],
        t_weld=4.0, h=0.01, Kp=120.0, Kd=24.0)
    smooth(s) = 3s^2 - 2s^3                    # S 曲线 (平滑加减速)

    function q_ref(t)
        s   = smooth(clamp(t / t_weld, 0.0, 1.0))
        pos = p_start .+ s .* (p_end .- p_start)
        return ik(arm, pos[1], pos[2])
    end
    qd_ref(t) = (q_ref(t + 1e-4) .- q_ref(t - 1e-4)) ./ 2e-4

    tau(q, v, t) = Kp.*(q_ref(t).-q) .+ Kd.*(qd_ref(t).-v) .+ gravity_comp(arm,q)
    L = (q, qd) -> lagrangian(arm, q, qd)
    ac= (q, v, t) -> arm_accel(arm, q, v, tau(q,v,t))

    q0 = q_ref(0.0);  v0 = qd_ref(0.0)
    t_arr, Q = del_run(L, q0, v0, t_weld+1.0, h; force=tau, acc=ac)
    tip = mapslices(q -> fk_tip(arm, q), Q; dims=2)
    ref = hcat([fk_tip(arm, q_ref(t)) for t in t_arr]...)'
    err = [norm(tip[k,:] .- ref[k,:]) for k in 1:size(tip,1)]
    return t_arr, tip, ref, err
end
