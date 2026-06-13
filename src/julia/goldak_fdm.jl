# goldak_fdm.jl — Goldak 双椭球 + 三维瞬态显式 FDM
# Julia 版本相较于 Python/NumPy 版本的优势:
#   • 内层时间步循环为原生 JIT 机器码, 无 Python 解释器开销
#   • 显式三重空间循环无需 NumPy pad/slice 技巧, 代码更清晰
#   • 典型加速比: 10–50× (取决于网格分辨率与步数)

"""
    run_goldak_fdm(; Q, eta, v_weld, t_end, x_start, kwargs...) -> (T, peak, meta)

三维瞬态显式有限差分, Goldak 双椭球热源, 半对称模型 (y≥0).
所有边界均为 edge-pad Neumann (零通量); 远场四面为 Dirichlet T0.
热源在每步做数值重归一化以精确满足能量守恒.

返回:
  T    :: Array{Float64,3}  — 终态温度场 [K]
  peak :: Array{Float64,3}  — 历史峰值温度 [K]
  meta :: NamedTuple         — (Nx,Ny,Nz,dx,T0,Tm)
"""
function run_goldak_fdm(;
        Q      = 8200.0,   # 电弧功率 [W]
        eta    = 0.8,      # 热效率
        v_weld = 8e-3,     # 焊接速度 [m/s]
        t_end  = 5.0,      # 仿真时长 [s]
        x_start= 0.015,    # 热源起始 x 坐标 [m]
        # Goldak 双椭球半轴 [m]
        a=4e-3, b=4e-3, cf=4e-3, cr=9e-3, ff=0.6,
        # 网格与工件
        Lx=0.10, Ly=0.025, Lz=0.020, dx=1.25e-3,
        # 材料 (低碳钢默认)
        rho=7850.0, cp=600.0, k_th=41.0,
        T0=298.0, Tm=1773.0)

    alpha   = k_th / (rho * cp)
    Nx, Ny, Nz = round(Int, Lx/dx), round(Int, Ly/dx), round(Int, Lz/dx)
    dt      = 0.4 * dx^2 / (6.0 * alpha)          # 显式稳定性上界
    n_steps = round(Int, t_end / dt)
    fr      = 2.0 - ff
    P_target = eta * Q / 2.0                       # 半模型吸收功率

    T    = fill(T0, Nx, Ny, Nz)
    peak = fill(T0, Nx, Ny, Nz)
    q_buf = similar(T)
    lap  = similar(T)

    for step in 1:n_steps
        xs = x_start + v_weld * (step - 1) * dt

        # ── 计算 Goldak 热源密度 ──────────────────────────────────
        q_sum = 0.0
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            xi  = (i-1)*dx - xs
            c   = xi >= 0 ? cf : cr
            f   = xi >= 0 ? ff : fr
            yy  = (j-1)*dx
            zz  = (k-1)*dx
            coef = 6.0*sqrt(3.0)*f*eta*Q / (a*b*c*pi^1.5)
            qv   = coef * exp(-3.0*(xi/c)^2 - 3.0*(yy/a)^2 - 3.0*(zz/b)^2)
            q_buf[i,j,k] = qv
            q_sum += qv
        end
        q_scale = q_sum*dx^3 > 1e-9 ? P_target/(q_sum*dx^3) : 0.0

        # ── Laplacian (edge-pad Neumann on all faces) ────────────
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            im = i==1  ? 1  : i-1;   ip = i==Nx ? Nx : i+1
            jm = j==1  ? 1  : j-1;   jp = j==Ny ? Ny : j+1
            km = k==1  ? 1  : k-1;   kp = k==Nz ? Nz : k+1
            lap[i,j,k] = T[ip,j,k] + T[im,j,k] +
                          T[i,jp,k] + T[i,jm,k] +
                          T[i,j,kp] + T[i,j,km] - 6.0*T[i,j,k]
        end

        # ── 更新温度 ──────────────────────────────────────────────
        coeff_lap = dt * alpha / dx^2
        coeff_q   = dt * q_scale / (rho * cp)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            T[i,j,k] += coeff_lap * lap[i,j,k] + coeff_q * q_buf[i,j,k]
        end

        # ── Dirichlet 远场四面 ────────────────────────────────────
        T[1,:,:]  .= T0;   T[end,:,:] .= T0
        T[:,end,:] .= T0;  T[:,:,end] .= T0

        # ── 峰值记录 ─────────────────────────────────────────────
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            peak[i,j,k] = max(peak[i,j,k], T[i,j,k])
        end
    end

    meta = (Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, T0=T0, Tm=Tm)
    return T, peak, meta
end


"""
    pool_size(T, meta) -> (L_mm, W_mm, D_mm)

从温度场估算熔池长/宽/深 [mm].
"""
function pool_size(T::Array{Float64,3}, meta)
    Tm, dx = meta.Tm, meta.dx
    melt = findall(T .>= Tm)
    isempty(melt) && return 0.0, 0.0, 0.0
    ix = [ci[1] for ci in melt]
    iy = [ci[2] for ci in melt]
    iz = [ci[3] for ci in melt]
    L = (maximum(ix) - minimum(ix)) * dx * 1e3
    W = 2.0 * maximum(iy) * dx * 1e3    # 半模型 → 全宽
    D = maximum(iz) * dx * 1e3
    return L, W, D
end
