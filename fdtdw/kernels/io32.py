import warp as wp
from .structs32 import EMState, EMStates, FieldsBuffer, TEMStates, DFTs, TEMStates_full


@wp.kernel
def save_checkpoint(
    state: EMState, checkpoints: EMStates, ckpt_idx_ptr: wp.array(dtype=int)
):
    i, j, k = wp.tid()

    t = ckpt_idx_ptr[0]

    checkpoints.Ex[t, i, j, k] = state.Ex[i, j, k]
    checkpoints.Ey[t, i, j, k] = state.Ey[i, j, k]
    checkpoints.Ez[t, i, j, k] = state.Ez[i, j, k]
    checkpoints.Hx[t, i, j, k] = state.Hx[i, j, k]
    checkpoints.Hy[t, i, j, k] = state.Hy[i, j, k]
    checkpoints.Hz[t, i, j, k] = state.Hz[i, j, k]

    checkpoints.psi_ex_y[t, i, j, k] = state.psi_ex_y[i, j, k]
    checkpoints.psi_ex_z[t, i, j, k] = state.psi_ex_z[i, j, k]
    checkpoints.psi_ey_x[t, i, j, k] = state.psi_ey_x[i, j, k]
    checkpoints.psi_ey_z[t, i, j, k] = state.psi_ey_z[i, j, k]
    checkpoints.psi_ez_x[t, i, j, k] = state.psi_ez_x[i, j, k]
    checkpoints.psi_ez_y[t, i, j, k] = state.psi_ez_y[i, j, k]

    checkpoints.psi_hx_y[t, i, j, k] = state.psi_hx_y[i, j, k]
    checkpoints.psi_hx_z[t, i, j, k] = state.psi_hx_z[i, j, k]
    checkpoints.psi_hy_x[t, i, j, k] = state.psi_hy_x[i, j, k]
    checkpoints.psi_hy_z[t, i, j, k] = state.psi_hy_z[i, j, k]
    checkpoints.psi_hz_x[t, i, j, k] = state.psi_hz_x[i, j, k]
    checkpoints.psi_hz_y[t, i, j, k] = state.psi_hz_y[i, j, k]


@wp.kernel
def load_checkpoint(
    state: EMState, checkpoints: EMStates, ckpt_idx_ptr: wp.array(dtype=wp.int32)
):
    i, j, k = wp.tid()
    t = ckpt_idx_ptr[0]

    state.Ex[i, j, k] = checkpoints.Ex[t, i, j, k]
    state.Ey[i, j, k] = checkpoints.Ey[t, i, j, k]
    state.Ez[i, j, k] = checkpoints.Ez[t, i, j, k]
    state.Hx[i, j, k] = checkpoints.Hx[t, i, j, k]
    state.Hy[i, j, k] = checkpoints.Hy[t, i, j, k]
    state.Hz[i, j, k] = checkpoints.Hz[t, i, j, k]

    state.psi_ex_y[i, j, k] = checkpoints.psi_ex_y[t, i, j, k]
    state.psi_ex_z[i, j, k] = checkpoints.psi_ex_z[t, i, j, k]
    state.psi_ey_x[i, j, k] = checkpoints.psi_ey_x[t, i, j, k]
    state.psi_ey_z[i, j, k] = checkpoints.psi_ey_z[t, i, j, k]
    state.psi_ez_x[i, j, k] = checkpoints.psi_ez_x[t, i, j, k]
    state.psi_ez_y[i, j, k] = checkpoints.psi_ez_y[t, i, j, k]

    state.psi_hx_y[i, j, k] = checkpoints.psi_hx_y[t, i, j, k]
    state.psi_hx_z[i, j, k] = checkpoints.psi_hx_z[t, i, j, k]
    state.psi_hy_x[i, j, k] = checkpoints.psi_hy_x[t, i, j, k]
    state.psi_hy_z[i, j, k] = checkpoints.psi_hy_z[t, i, j, k]
    state.psi_hz_x[i, j, k] = checkpoints.psi_hz_x[t, i, j, k]
    state.psi_hz_y[i, j, k] = checkpoints.psi_hz_y[t, i, j, k]


@wp.kernel
def load_dft_magnitude(dfts: DFTs, state: EMState, freq_idx: int):
    i, j, k = wp.tid()

    state.Ex[i, j, k] = wp.length(dfts.Ex[freq_idx, i, j, k])
    state.Ey[i, j, k] = wp.length(dfts.Ey[freq_idx, i, j, k])
    state.Ez[i, j, k] = wp.length(dfts.Ez[freq_idx, i, j, k])

    state.Hx[i, j, k] = wp.length(dfts.Hx[freq_idx, i, j, k])
    state.Hy[i, j, k] = wp.length(dfts.Hy[freq_idx, i, j, k])
    state.Hz[i, j, k] = wp.length(dfts.Hz[freq_idx, i, j, k])


@wp.kernel
def clear_state(state: EMState):
    i, j, k = wp.tid()

    state.Ex[i, j, k] = 0.0
    state.Ey[i, j, k] = 0.0
    state.Ez[i, j, k] = 0.0
    state.Hx[i, j, k] = 0.0
    state.Hy[i, j, k] = 0.0
    state.Hz[i, j, k] = 0.0

    state.psi_ex_y[i, j, k] = 0.0
    state.psi_ex_z[i, j, k] = 0.0
    state.psi_ey_x[i, j, k] = 0.0
    state.psi_ey_z[i, j, k] = 0.0
    state.psi_ez_x[i, j, k] = 0.0
    state.psi_ez_y[i, j, k] = 0.0

    state.psi_hx_y[i, j, k] = 0.0
    state.psi_hx_z[i, j, k] = 0.0
    state.psi_hy_x[i, j, k] = 0.0
    state.psi_hy_z[i, j, k] = 0.0
    state.psi_hz_x[i, j, k] = 0.0
    state.psi_hz_y[i, j, k] = 0.0


@wp.kernel
def save_field(state: EMState, buffer: FieldsBuffer, pos_ptr: wp.array(dtype=int)):
    i, j, k = wp.tid()

    pos = pos_ptr[0]

    buffer.Ex[pos, i, j, k] = state.Ex[i, j, k]
    buffer.Ey[pos, i, j, k] = state.Ey[i, j, k]
    buffer.Ez[pos, i, j, k] = state.Ez[i, j, k]
    buffer.Hx[pos, i, j, k] = state.Hx[i, j, k]
    buffer.Hy[pos, i, j, k] = state.Hy[i, j, k]
    buffer.Hz[pos, i, j, k] = state.Hz[i, j, k]


@wp.kernel
def save_detector_yz(
    state: EMState,
    detector: TEMStates,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + normal_offset
    gy = detector.OFFSETY + u
    gz = detector.OFFSETZ + v

    detector.Eu[t, u, v] = state.Ey[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ez[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hy[gx, gy, gz]
    detector.Hv[t, u, v] = state.Hz[gx, gy, gz]


@wp.kernel
def save_detector_xz(
    state: EMState,
    detector: TEMStates,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + u
    gy = detector.OFFSETY + normal_offset
    gz = detector.OFFSETZ + v

    detector.Eu[t, u, v] = state.Ex[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ez[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hx[gx, gy, gz]
    detector.Hv[t, u, v] = state.Hz[gx, gy, gz]


@wp.kernel
def save_detector_xy(
    state: EMState,
    detector: TEMStates,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + u
    gy = detector.OFFSETY + v
    gz = detector.OFFSETZ + normal_offset

    detector.Eu[t, u, v] = state.Ex[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ey[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hx[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hx[gx, gy, gz]


@wp.kernel
def save_detector_full_yz(
    state: EMState,
    detector: TEMStates_full,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + normal_offset
    gy = detector.OFFSETY + u
    gz = detector.OFFSETZ + v

    detector.Eu[t, u, v] = state.Ey[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ez[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hy[gx, gy, gz]
    detector.Hv[t, u, v] = state.Hz[gx, gy, gz]
    detector.Hu_n[t, u, v] = state.Hy[gx + 1, gy, gz]
    detector.Hv_n[t, u, v] = state.Hz[gx + 1, gy, gz]


@wp.kernel
def save_detector_full_xz(
    state: EMState,
    detector: TEMStates_full,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + u
    gy = detector.OFFSETY + normal_offset
    gz = detector.OFFSETZ + v

    detector.Eu[t, u, v] = state.Ex[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ez[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hx[gx, gy, gz]
    detector.Hv[t, u, v] = state.Hz[gx, gy, gz]
    detector.Hu_n[t, u, v] = state.Hx[gx, gy + 1, gz]
    detector.Hv_n[t, u, v] = state.Hz[gx, gy + 1, gz]


@wp.kernel
def save_detector_full_xy(
    state: EMState,
    detector: TEMStates_full,
    time_ptr: wp.array(dtype=int),
    normal_offset: int,
):
    u, v = wp.tid()
    t = time_ptr[0]

    gx = detector.OFFSETX + u
    gy = detector.OFFSETY + v
    gz = detector.OFFSETZ + normal_offset

    detector.Eu[t, u, v] = state.Ex[gx, gy, gz]
    detector.Ev[t, u, v] = state.Ey[gx, gy, gz]
    detector.Hu[t, u, v] = state.Hx[gx, gy, gz]
    detector.Hv[t, u, v] = state.Hy[gx, gy, gz]
    detector.Hu_n[t, u, v] = state.Hx[gx, gy, gz + 1]
    detector.Hv_n[t, u, v] = state.Hy[gx, gy, gz + 1]
