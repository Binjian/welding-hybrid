# contact_vi.jl — 非光滑接触变分模型 (CMT 触池循环 / 弹性反冲基准)
# 自由相用辛 Verlet; 接触事件用二分精确定位 + 变分碰撞映射
# 对比: 罚函数法在刚性接触下产生巨量虚假能量注入

struct ContactParams
    m      ::Float64  # 熔滴质量 [kg]
    k      ::Float64  # 弹簧刚度 [N/m]
    c      ::Float64  # 阻尼 [N·s/m]
    gap    ::Float64  # 熔池表面位置 [m]
    v_feed ::Float64  # 送丝速度 [m/s]
    v_retr ::Float64  # 回抽速度 [m/s]
    F_rup  ::Float64  # 断桥张力阈值 [N]
end

function ContactParams(; rd=0.5e-3, gamma=1.2, rho=7000.0, zeta=0.03,
                         gap=0.4e-3, v_feed=0.08, v_retract=0.10,
                         F_rup_frac=1.4)
    m    = rho*4/3*pi*rd^3
    k    = 32*pi*gamma/3
    c    = 2*zeta*sqrt(k*m)
    F_rup= F_rup_frac * k * gap
    ContactParams(m, k, c, gap, v_feed, v_retract, F_rup)
end

# ── 单步辛 Verlet (自由相) ────────────────────────────────────────
function _verlet_step(m, k, c, x_eq, x, v, h)
    a0    = (-k*(x-x_eq) - c*v)/m
    v_h   = v + 0.5h*a0
    x_new = x + h*v_h
    a1    = (-k*(x_new-x_eq) - c*v_h)/m
    v_new = v_h + 0.5h*a1
    return x_new, v_new
end

# ── 二分事件定位 ─────────────────────────────────────────────────
function _bisect_contact(m, k, c, x_eq, x, v, h, threshold)
    lo, hi = 0.0, h
    for _ in 1:50
        mid = 0.5*(lo+hi)
        # 简单线性预测 (步长很小时精度足够)
        x_mid = x + mid*v
        if x_mid >= threshold
            hi = mid
        else
            lo = mid
        end
        hi - lo < 1e-13 && break
    end
    dt1 = hi
    v_imp = v + dt1*(-k*(x-x_eq)-c*v)/m   # 近似撞击速度
    return dt1, v_imp
end

"""
    cmt_cycle(p; t_end, dt) -> (t_arr, x_arr, x_eq_arr, phase_arr, events)

CMT 机械振荡循环: 自由相辛 Verlet, 触池事件二分定位, 断桥后状态重置.
"""
function cmt_cycle(p::ContactParams; t_end=0.06, dt=2e-6)
    n     = round(Int, t_end/dt)
    t_arr = zeros(n);  x_arr = zeros(n)
    xeq_arr=zeros(n);  ph_arr = zeros(n)

    x, v, x_eq, phase = 0.0, 0.0, 0.0, 0
    events = Tuple{Float64,Symbol}[]

    for i in 1:n
        t = (i-1)*dt
        if phase == 0          # ── 自由相 (送丝推进) ──────────
            x_eq += p.v_feed*dt
            x_new, v_new = _verlet_step(p.m, p.k, p.c, x_eq, x, v, dt)
            if x_new >= p.gap
                dt1, v_imp = _bisect_contact(p.m, p.k, p.c, x_eq, x, v, dt, p.gap)
                x, v = p.gap, 0.0    # e=0 湿接触
                phase = 1
                push!(events, (t+dt1, :dip))
            else
                x, v = x_new, v_new
            end
        else                   # ── 附着相 (回抽) ───────────────
            x_eq -= p.v_retr*dt
            if p.k*(p.gap - x_eq) >= p.F_rup
                x, v, phase = x_eq, 0.0, 0
                push!(events, (t, :rupture))
            end
        end
        t_arr[i], x_arr[i], xeq_arr[i], ph_arr[i] = t, x, x_eq, phase
    end
    return t_arr, x_arr, xeq_arr, ph_arr, events
end

"""
    bounce_nonsmooth(p; t_end, dt, e, x0, Fc) -> (t, x, E)

弹性反冲基准 (恒力推向壁面 x=0): 非光滑 VI, 精确事件定位 + 变分碰撞映射.
能量仅在碰撞事件处按物理规律 e² 衰减.
"""
function bounce_nonsmooth(p::ContactParams; t_end=0.03, dt=4e-6,
                           e=0.85, x0=-4e-4, Fc=nothing)
    Fc = Fc === nothing ? 0.5*p.k*p.gap : Fc
    n  = round(Int, t_end/dt)
    T  = zeros(n);  X = zeros(n);  E = zeros(n)
    x, v = x0, 0.0
    for i in 1:n
        t = (i-1)*dt
        a = Fc/p.m
        v_h   = v + 0.5dt*a
        x_new = x + dt*v_h
        if x_new >= 0.0
            # 二分精确定位撞击时刻
            lo, hi = 0.0, dt
            for _ in 1:50
                mid = 0.5*(lo+hi)
                (x + mid*v_h) >= 0.0 ? (hi=mid) : (lo=mid)
                hi-lo < 1e-13 && break
            end
            dt1   = hi
            v_imp = v_h                       # 撞击时刻半步速度
            v_r   = -e*v_imp                  # 变分碰撞映射
            x     = 0.0
            dt2   = dt - dt1
            v_h2  = v_r + 0.5dt2*a
            x     = x + dt2*v_h2
            v     = v_h2 + 0.5dt2*a
        else
            v     = v_h + 0.5dt*a
            x     = x_new
        end
        T[i], X[i] = t, x
        E[i] = 0.5*p.m*v^2 - Fc*min(x, 0.0)
    end
    return T, X, E
end

"""
    bounce_penalty(p; t_end, dt, k_pen_frac, x0, Fc) -> (t, x, E)

罚函数法基线: 相同步长但不做事件定位, 接触刚度欠解析 → 虚假能量注入.
"""
function bounce_penalty(p::ContactParams; t_end=0.03, dt=4e-6,
                         k_pen_frac=2.5e4, x0=-4e-4, Fc=nothing)
    Fc    = Fc === nothing ? 0.5*p.k*p.gap : Fc
    k_pen = k_pen_frac * p.k
    n     = round(Int, t_end/dt)
    T     = zeros(n);  X = zeros(n);  E = zeros(n)
    x, v  = x0, 0.0
    for i in 1:n
        t  = (i-1)*dt
        F  = Fc - (x > 0.0 ? k_pen*x : 0.0)
        a  = F/p.m
        v_h= v + 0.5dt*a
        x  = x + dt*v_h
        F1 = Fc - (x > 0.0 ? k_pen*x : 0.0)
        v  = v_h + 0.5dt*(F1/p.m)
        T[i], X[i] = t, clamp(x,-2e-3,2e-3)
        E[i] = 0.5*p.m*v^2 - Fc*min(x,0.0) + 0.5*k_pen*max(x,0.0)^2
    end
    return T, X, E
end
