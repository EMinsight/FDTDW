import warp as wp
from .structs32 import DFTs, EMState, Gradients, TEMStates, TEMDFTs, TEMStates_full


@wp.kernel
def compute_dft_coeffs(
    freqs: wp.array(dtype=float), dt: float, coeffs: wp.array2d(dtype=wp.vec2)
):
    n, nf = wp.tid()

    w = freqs[nf]

    theta = w * dt * float(n)

    real_part = wp.cos(theta) * dt
    imag_part = -wp.sin(theta) * dt

    coeffs[n, nf] = wp.vec2(real_part, imag_part)


@wp.kernel
def compute_dft_pdf(
    signal: wp.array(dtype=float),
    coeffs: wp.array2d(dtype=wp.vec2),
    pd_freqs: wp.array(dtype=float),
):
    nf = wp.tid()

    sum_real = float(0.0)
    sum_imag = float(0.0)

    num_timesteps = signal.shape[0]

    for n in range(num_timesteps):
        val = signal[n]

        c = coeffs[n, nf]

        sum_real += val * c[0]
        sum_imag += val * c[1]

    pd_freqs[nf] = sum_real * sum_real + sum_imag * sum_imag


@wp.kernel
def accumulate_dft(
    coeffs: wp.array2d(dtype=wp.vec2),
    state: EMState,
    dfts: DFTs,
    idx_t: wp.array(dtype=int),
):
    i, j, k = wp.tid()
    num_freqs = coeffs.shape[1]
    n = idx_t[0]
    for f in range(num_freqs):
        c = coeffs[n, f]

        dfts.Ex[f, i, j, k] = dfts.Ex[f, i, j, k] + c * state.Ex[i, j, k]
        dfts.Ey[f, i, j, k] = dfts.Ey[f, i, j, k] + c * state.Ey[i, j, k]
        dfts.Ez[f, i, j, k] = dfts.Ez[f, i, j, k] + c * state.Ez[i, j, k]

        dfts.Hx[f, i, j, k] = dfts.Hx[f, i, j, k] + c * state.Hx[i, j, k]
        dfts.Hy[f, i, j, k] = dfts.Hy[f, i, j, k] + c * state.Hy[i, j, k]
        dfts.Hz[f, i, j, k] = dfts.Hz[f, i, j, k] + c * state.Hz[i, j, k]


@wp.kernel
def accumulate_dft_4d(
    coeffs: wp.array2d(dtype=wp.vec2),
    state: EMState,
    dfts: DFTs,
    idx_t: wp.array(dtype=int),
):
    f, i, j, k = wp.tid()
    n = idx_t[0]
    c = coeffs[n, f]

    dfts.Ex[f, i, j, k] = dfts.Ex[f, i, j, k] + c * state.Ex[i, j, k]
    dfts.Ey[f, i, j, k] = dfts.Ey[f, i, j, k] + c * state.Ey[i, j, k]
    dfts.Ez[f, i, j, k] = dfts.Ez[f, i, j, k] + c * state.Ez[i, j, k]

    dfts.Hx[f, i, j, k] = dfts.Hx[f, i, j, k] + c * state.Hx[i, j, k]
    dfts.Hy[f, i, j, k] = dfts.Hy[f, i, j, k] + c * state.Hy[i, j, k]
    dfts.Hz[f, i, j, k] = dfts.Hz[f, i, j, k] + c * state.Hz[i, j, k]


@wp.kernel
def compute_tem_dft(
    coeffs: wp.array2d(dtype=wp.vec2), history: TEMStates, dfts: TEMDFTs
):
    u, v = wp.tid()

    num_timesteps = coeffs.shape[0]
    num_freqs = coeffs.shape[1]

    for f in range(num_freqs):

        sum_Eu = wp.vec2(0.0, 0.0)
        sum_Ev = wp.vec2(0.0, 0.0)
        sum_Hu = wp.vec2(0.0, 0.0)
        sum_Hv = wp.vec2(0.0, 0.0)

        for t in range(num_timesteps):
            c = coeffs[t, f]

            val_Eu = history.Eu[t, u, v]
            val_Ev = history.Ev[t, u, v]
            val_Hu = history.Hu[t, u, v]
            val_Hv = history.Hv[t, u, v]

            sum_Eu += c * val_Eu
            sum_Ev += c * val_Ev
            sum_Hu += c * val_Hu
            sum_Hv += c * val_Hv

        dfts.Eu[f, u, v] = sum_Eu
        dfts.Ev[f, u, v] = sum_Ev
        dfts.Hu[f, u, v] = sum_Hu
        dfts.Hv[f, u, v] = sum_Hv


@wp.func
def complex_mul(a: wp.vec2, b: wp.vec2):
    return wp.vec2(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def complex_geometric_mean(z1: wp.vec2, z2: wp.vec2):

    P = complex_mul(z1, z2)

    mag_P = wp.length(P)

    if mag_P < 1e-12:
        return wp.vec2(0.0, 0.0)

    theta_P = wp.atan2(P[1], P[0])
    sqrt_mag = wp.sqrt(mag_P)

    root = wp.vec2(sqrt_mag * wp.cos(0.5 * theta_P), sqrt_mag * wp.sin(0.5 * theta_P))

    if wp.dot(root, z1) < 0.0:
        root = -root

    return root


@wp.kernel
def compute_collocated_tem_dft(
    coeffs: wp.array2d(dtype=wp.vec2), history: TEMStates_full, dfts: TEMDFTs
):
    u, v = wp.tid()

    num_timesteps = coeffs.shape[0]
    num_freqs = coeffs.shape[1]

    for f in range(num_freqs):

        sum_Eu = wp.vec2(0.0, 0.0)
        sum_Ev = wp.vec2(0.0, 0.0)

        sum_Hu = wp.vec2(0.0, 0.0)
        sum_Hu_n = wp.vec2(0.0, 0.0)

        sum_Hv = wp.vec2(0.0, 0.0)
        sum_Hv_n = wp.vec2(0.0, 0.0)

        for t in range(num_timesteps):
            c = coeffs[t, f]

            val_Eu = history.Eu[t, u, v]
            val_Ev = history.Ev[t, u, v]

            val_Hu = history.Hu[t, u, v]
            val_Hu_n = history.Hu_n[t, u, v]

            val_Hv = history.Hv[t, u, v]
            val_Hv_n = history.Hv_n[t, u, v]

            sum_Eu += c * val_Eu
            sum_Ev += c * val_Ev

            sum_Hu += c * val_Hu
            sum_Hu_n += c * val_Hu_n

            sum_Hv += c * val_Hv
            sum_Hv_n += c * val_Hv_n

        dfts.Eu[f, u, v] = sum_Eu
        dfts.Ev[f, u, v] = sum_Ev

        dfts.Hu[f, u, v] = complex_geometric_mean(sum_Hu, sum_Hu_n)
        dfts.Hv[f, u, v] = complex_geometric_mean(sum_Hv, sum_Hv_n)


@wp.func
def complex_mul_real(a: wp.vec2, b: wp.vec2):
    return a[0] * b[0] - a[1] * b[1]


@wp.kernel
def compute_gradients(
    dfts: DFTs, dfts_adj: DFTs, w_freqs: wp.array(dtype=wp.float32), grads: Gradients
):
    f, i, j, k = wp.tid()

    val_Ex = w_freqs[f] * complex_mul_real(dfts.Ex[f, i, j, k], dfts_adj.Ex[f, i, j, k])
    val_Ey = w_freqs[f] * complex_mul_real(dfts.Ey[f, i, j, k], dfts_adj.Ey[f, i, j, k])
    val_Ez = w_freqs[f] * complex_mul_real(dfts.Ez[f, i, j, k], dfts_adj.Ez[f, i, j, k])

    wp.atomic_add(grads.grad_CE_x, i, j, k, val_Ex)
    wp.atomic_add(grads.grad_CE_y, i, j, k, val_Ey)
    wp.atomic_add(grads.grad_CE_z, i, j, k, val_Ez)

    val_Hx = w_freqs[f] * complex_mul_real(dfts.Hx[f, i, j, k], dfts_adj.Hx[f, i, j, k])
    val_Hy = w_freqs[f] * complex_mul_real(dfts.Hy[f, i, j, k], dfts_adj.Hy[f, i, j, k])
    val_Hz = w_freqs[f] * complex_mul_real(dfts.Hz[f, i, j, k], dfts_adj.Hz[f, i, j, k])

    wp.atomic_add(grads.grad_CH_x, i, j, k, val_Hx)
    wp.atomic_add(grads.grad_CH_y, i, j, k, val_Hy)
    wp.atomic_add(grads.grad_CH_z, i, j, k, val_Hz)
