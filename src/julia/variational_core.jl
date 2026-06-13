# variational_core.jl — 变分积分器核心
#
# ForcedVerlet  : 辛 Stormer-Verlet + 离散 Lagrange-d'Alembert 强迫 (显式, 二阶)
# MidpointDEL   : 中点离散 Euler-Lagrange (隐式, 二阶)
#                 Newton 求解器使用 ForwardDiff.jacobian — 精确 Jacobian,
#                 无有限差分噪声, 收敛速度远快于 Python scipy.fsolve 版本
# RK4           : 非辛基线参照与 DEL 启动子步

# ─────────────────────────────────────────────────────────────────────
# 辅助: RK4 单步 (向量化)
# ─────────────────────────────────────────────────────────────────────
function rk4_step(acc, q::AbstractVector, v::AbstractVector, t::Float64, h::Float64)
    k1q = v;                   k1v = acc(q,            v,            t)
    k2q = v .+ 0.5h.*k1v;     k2v = acc(q.+0.5h.*k1q, v.+0.5h.*k1v, t+0.5h)
    k3q = v .+ 0.5h.*k2v;     k3v = acc(q.+0.5h.*k2q, v.+0.5h.*k2v, t+0.5h)
    k4q = v .+    h.*k3v;     k4v = acc(q.+   h.*k3q, v.+   h.*k3v, t+h)
    q_new = q .+ (h/6).*(k1q .+ 2k2q .+ 2k3q .+ k4q)
    v_new = v .+ (h/6).*(k1v .+ 2k2v .+ 2k3v .+ k4v)
    return q_new, v_new
end

function rk4_run(acc, q0, v0, t_end, h)
    n = round(Int, t_end / h)
    Q = zeros(n+1, length(q0))
    V = zeros(n+1, length(q0))
    Q[1,:] = q0;  V[1,:] = v0
    q, v, t = copy(q0), copy(v0), 0.0
    for k in 1:n
        q, v = rk4_step(acc, q, v, t, h)
        Q[k+1,:] = q;  V[k+1,:] = v
        t += h
    end
    return h .* (0:n), Q, V
end

# ─────────────────────────────────────────────────────────────────────
# ForcedVerlet (显式辛 Verlet + DLA 强迫)
# ─────────────────────────────────────────────────────────────────────
"""
    forced_verlet_run(m, grad_V, force, q0, v0, t_end, h) -> (t, Q, V)

m       : 质量 (标量或向量)
grad_V  : q -> ∇V(q)
force   : (q, v, t) -> 广义力向量
"""
function forced_verlet_run(m, grad_V, force, q0, v0, t_end::Float64, h::Float64)
    n = round(Int, t_end / h)
    ndof = length(q0)
    T_arr = h .* Float64.(0:n)
    Q = zeros(n+1, ndof);  V = zeros(n+1, ndof)
    Q[1,:] = q0;  V[1,:] = v0
    q, v, t = copy(Float64.(q0)), copy(Float64.(v0)), 0.0
    for k in 1:n
        F0    = .-grad_V(q) .+ force(q, v, t)
        v_half = v .+ (0.5h) .* (F0 ./ m)
        q_new  = q .+ h .* v_half
        F1    = .-grad_V(q_new) .+ force(q_new, v_half, t+h)
        v_new  = v_half .+ (0.5h) .* (F1 ./ m)
        q, v, t = q_new, v_new, t+h
        Q[k+1,:] = q;  V[k+1,:] = v
    end
    return T_arr, Q, V
end

# ─────────────────────────────────────────────────────────────────────
# MidpointDEL — 单步 Newton (ForwardDiff 精确 Jacobian)
# ─────────────────────────────────────────────────────────────────────
"""
    del_step(L, q_prev, q, h; force, t) -> q_next

离散 Euler-Lagrange 方程的 Newton 求解:
  D₂Lᵈ(qₖ₋₁,qₖ) + D₁Lᵈ(qₖ,qₖ₊₁)
  + h/2·F⁻(midₖ₋₁) + h/2·F⁺(midₖ) = 0

ForwardDiff.jacobian 提供精确 Jacobian, 每次 Newton 迭代收敛更快.
"""
function del_step(L, q_prev::AbstractVector, q::AbstractVector,
                  h::Float64; force=nothing, t::Float64=0.0)
    Ld(q0, q1)  = h * L(0.5.*(q0.+q1), (q1.-q0)./h)
    D2Ld(q0,q1) = ForwardDiff.gradient(x -> Ld(q0, x), q1)
    D1Ld(q0,q1) = ForwardDiff.gradient(x -> Ld(x,  q1), q0)

    base = D2Ld(q_prev, q)
    if force !== nothing
        v_prev = (q .- q_prev) ./ h
        base   = base .+ (0.5h) .* force(0.5.*(q_prev.+q), v_prev, t - 0.5h)
    end

    # 残差函数 (ForwardDiff 会对此做 AD)
    function residual(q_next::AbstractVector)
        r = base .+ D1Ld(q, q_next)
        if force !== nothing
            v_n = (q_next .- q) ./ h
            r   = r .+ (0.5h) .* force(0.5.*(q.+q_next), v_n, t + 0.5h)
        end
        return r
    end

    x = 2.0.*q .- q_prev          # 线性外插初值
    for _ in 1:30
        r = residual(x)
        maximum(abs, r) < 1e-12 && break
        J = ForwardDiff.jacobian(residual, x)
        x = x .- J \ r
    end
    return x
end

"""
    del_run(L, q0, v0, t_end, h; force, acc) -> (t_arr, Q)

完整 DEL 积分: RK4 细步生成 q₁ 启动, 然后递推.
"""
function del_run(L, q0, v0, t_end::Float64, h::Float64;
                 force=nothing, acc=nothing)
    n = round(Int, t_end / h)
    Q = zeros(n+1, length(q0))
    Q[1,:] = q0

    # 启动 q₁: 用 acc RK4 细步 (sub=20 步)
    local_acc = acc !== nothing ? acc :
        (q,v,t) -> begin
            g  = ForwardDiff.gradient(x -> -L(x, v), q)   # fallback
            gg = ForwardDiff.gradient(x -> L(q, x), v)
            # 简单近似: 用 Lagrangian 梯度 (仅限对角质量矩阵场景)
            g
        end
    q_bk, v_bk = copy(Float64.(q0)), copy(Float64.(v0))
    sub = 20;  hs = h / sub
    for _ in 1:sub
        q_bk, v_bk = rk4_step(local_acc, q_bk, v_bk, 0.0, hs)
    end
    Q[2,:] = q_bk

    t = h
    for k in 2:n
        Q[k+1,:] = del_step(L, Q[k-1,:], Q[k,:], h; force=force, t=t)
        t += h
    end
    return h .* Float64.(0:n), Q
end
