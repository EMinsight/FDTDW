import warp as wp
from .structs32 import EMState, Properties


@wp.kernel
def update_yee_h(state: EMState, props: Properties):
    i, j, k = wp.tid()
    nx = state.Ex.shape[0]
    ny = state.Ex.shape[1]
    nz = state.Ex.shape[2]

    if i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    dEz_dy = state.Ez[i, j + 1, k] - state.Ez[i, j, k]
    dEy_dz = state.Ey[i, j, k + 1] - state.Ey[i, j, k]

    dEx_dz = state.Ex[i, j, k + 1] - state.Ex[i, j, k]
    dEz_dx = state.Ez[i + 1, j, k] - state.Ez[i, j, k]

    dEy_dx = state.Ey[i + 1, j, k] - state.Ey[i, j, k]
    dEx_dy = state.Ex[i, j + 1, k] - state.Ex[i, j, k]



    state.Hx[i, j, k] = props.CHA_X[i, j, k] * state.Hx[i, j, k] - props.CHB_X[
        i, j, k
    ] * (dEz_dy - dEy_dz)
    state.Hy[i, j, k] = props.CHA_Y[i, j, k] * state.Hy[i, j, k] - props.CHB_Y[
        i, j, k
    ] * (dEx_dz - dEz_dx)
    state.Hz[i, j, k] = props.CHA_Z[i, j, k] * state.Hz[i, j, k] - props.CHB_Z[
        i, j, k
    ] * (dEy_dx - dEx_dy)


@wp.kernel
def update_yee_e(state: EMState, props: Properties):
    i, j, k = wp.tid()
    nx = state.Ex.shape[0]
    ny = state.Ex.shape[1]
    nz = state.Ex.shape[2]

    if i == 0 or j == 0 or k == 0 or i >= nx or j >= ny or k >= nz:
        return

    dHz_dy = state.Hz[i, j, k] - state.Hz[i, j - 1, k]
    dHy_dz = state.Hy[i, j, k] - state.Hy[i, j, k - 1]

    dHx_dz = state.Hx[i, j, k] - state.Hx[i, j, k - 1]
    dHz_dx = state.Hz[i, j, k] - state.Hz[i - 1, j, k]

    dHy_dx = state.Hy[i, j, k] - state.Hy[i - 1, j, k]
    dHx_dy = state.Hx[i, j, k] - state.Hx[i, j - 1, k]


    state.Ex[i, j, k] = props.CEA_X[i, j, k] * state.Ex[i, j, k] + props.CEB_X[
        i, j, k
    ] * (dHz_dy - dHy_dz)
    state.Ey[i, j, k] = props.CEA_Y[i, j, k] * state.Ey[i, j, k] + props.CEB_Y[
        i, j, k
    ] * (dHx_dz - dHz_dx)
    state.Ez[i, j, k] = props.CEA_Z[i, j, k] * state.Ez[i, j, k] + props.CEB_Z[
        i, j, k
    ] * (dHy_dx - dHx_dy)
