# droplet_vi.jl — 熔滴振荡 / 脉冲 MIG 共振 (变分积分器)
# 悬垂熔滴 Rayleigh l=2 模态: k = 32πγ/3, f0 = √(k/m)/(2π)
# ForcedVerlet 比隐式 Euler 保真共振峰, 避免人工数值阻尼误导脉冲参数整定

struct DropletParams
    m   :: Float64    # 质量 [kg]
    k   :: Float64    # 等效刚度 [N/m]
    c   :: Float64    # 阻尼 [N·s/m]
    F0  :: Float64    # 脉冲力幅 [N]
    f0  :: Float64    # 固有频率 [Hz]
    duty:: Float64
end

function DropletParams(; rd=0.5e-3, gamma=1.2, rho=7000.0, zeta=0.02,
                         F0_frac=0.15, duty=0.35)
    m  = rho * 4/3*pi*rd^3
    k  = 32*pi*gamma/3
    f0 = sqrt(k/m)/(2pi)
    c  = 2*zeta*sqrt(k*m)
    F0 = F0_frac * k * rd
    DropletParams(m, k, c, F0, f0, duty)
end

_pulse(F0, duty, fp, t) = (t*fp) % 1.0 < duty ? F0 : 0.0

# ── 变分积分器 (ForcedVerlet) ─────────────────────────────────────
function droplet_vi_run(p::DropletParams, fp::Float64,
                        t_end::Float64, h::Float64;
                        x0=0.0, v0=0.0)
    grad_V = q -> p.k .* q
    force  = (q, v, t) -> SA[-p.c*v[1] + _pulse(p.F0, p.duty, fp, t)]
    t_arr, Q, V = forced_verlet_run(p.m, grad_V, force,
                                    SA[x0], SA[v0], t_end, h)
    return t_arr, Q[:,1], V[:,1]
end

# ── 隐式 Euler 基线 ────────────────────────────────────────────────
function droplet_implicit_euler(p::DropletParams, fp::Float64,
                                t_end::Float64, h::Float64;
                                x0=0.0, v0=0.0)
    n = round(Int, t_end/h)
    X = zeros(n+1)
    X[1] = x0;  x, v = x0, v0
    a11, a12 = 1.0, -h
    a21, a22 = h*p.k/p.m, 1.0 + h*p.c/p.m
    det = a11*a22 - a12*a21
    for i in 1:n
        F = _pulse(p.F0, p.duty, fp, i*h) / p.m
        b1, b2 = x, v + h*F
        x = (b1*a22 - a12*b2)/det
        v = (a11*b2 - b1*a21)/det
        X[i+1] = x
    end
    return h .* Float64.(0:n), X
end

"""
    droplet_resonance_sweep(p, fp_arr; method, periods, steps) -> amplitudes

扫描脉冲频率, 返回稳态振幅 [m].
"""
function droplet_resonance_sweep(p::DropletParams, fp_arr::Vector{Float64};
                                 method::String="vi", periods::Int=60,
                                 steps_per_period::Int=22)
    amps = similar(fp_arr)
    T0   = 1.0 / p.f0
    h    = T0 / steps_per_period
    for (i, fp) in enumerate(fp_arr)
        t_end = periods * T0
        if method == "vi"
            _, X, _ = droplet_vi_run(p, fp, t_end, h)
        else
            _, X    = droplet_implicit_euler(p, fp, t_end, h)
        end
        tail = X[round(Int, 0.7*length(X)):end]
        amps[i] = 0.5*(maximum(tail) - minimum(tail))
    end
    return amps
end

"""
    droplet_free_energy(p, t_end, h) -> (t, E_vi, E_ee, E_ie)

自由振荡 (zeta=0) 三种积分器能量对比.
"""
function droplet_free_energy(p::DropletParams, t_end::Float64, h::Float64;
                              x0=1e-4)
    p0 = DropletParams(p.m, p.k, 0.0, p.F0, p.f0, p.duty)
    nofp = 0.0                      # 零脉冲频率

    # VI
    t, X_vi, V_vi = droplet_vi_run(p0, nofp, t_end, h; x0=x0)
    E_vi = @. 0.5*p0.m*V_vi^2 + 0.5*p0.k*X_vi^2

    # Implicit Euler (能量用速度中心差分)
    t_ie, X_ie = droplet_implicit_euler(p0, nofp, t_end, h; x0=x0)
    V_ie = [i==1||i==length(X_ie) ? 0.0 : (X_ie[i+1]-X_ie[i-1])/(2h)
            for i in eachindex(X_ie)]
    E_ie = @. 0.5*p0.m*V_ie^2 + 0.5*p0.k*X_ie^2

    # Explicit Euler
    n   = round(Int, t_end/h)
    X_ee= zeros(n+1);  X_ee[1]=x0;  x,v = x0, 0.0
    for i in 1:n
        ax  = -p0.k*x/p0.m
        x, v = x+h*v, v+h*ax
        X_ee[i+1] = x
    end
    V_ee = [i==1||i==length(X_ee) ? 0.0 : (X_ee[i+1]-X_ee[i-1])/(2h)
            for i in eachindex(X_ee)]
    E_ee = @. 0.5*p0.m*V_ee^2 + 0.5*p0.k*X_ee^2

    return t, E_vi, E_ee[1:length(t)], E_ie[1:length(t)]
end
