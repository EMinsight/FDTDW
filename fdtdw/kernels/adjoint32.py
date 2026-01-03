import warp as wp
from .structs32 import EMState, FieldsBuffer, Gradients, TEMStates


@wp.kernel
def calc_grad(
    grads: Gradients,
    fields_fwd: FieldsBuffer,
    state_adj: EMState,
    pos_ptr: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    pos = pos_ptr[0]

    grads.grad_CE_x[i, j, k] += fields_fwd.Ex[pos, i, j, k] * state_adj.Ex[i, j, k]
    grads.grad_CE_y[i, j, k] += fields_fwd.Ey[pos, i, j, k] * state_adj.Ey[i, j, k]
    grads.grad_CE_z[i, j, k] += fields_fwd.Ez[pos, i, j, k] * state_adj.Ez[i, j, k]

    grads.grad_CH_x[i, j, k] += fields_fwd.Hx[pos, i, j, k] * state_adj.Hx[i, j, k]
    grads.grad_CH_y[i, j, k] += fields_fwd.Hy[pos, i, j, k] * state_adj.Hy[i, j, k]
    grads.grad_CH_z[i, j, k] += fields_fwd.Hz[pos, i, j, k] * state_adj.Hz[i, j, k]


# @wp.kernel
# def map_yz(
#     history: TEMStates,
#     schedule: TEMStates,
# ):
#     t, u, v = wp.tid()
#
#     Ey = history.Eu[t, u, v]
#     Ez = history.Ev[t, u, v]
#     Hy = history.Hu[t, u, v]
#     Hz = history.Hv[t, u, v]
#
#     schedule.Eu[t, u, v] = Hz
#     schedule.Ev[t, u, v] = -Hy
#     schedule.Hu[t, u, v] = -Ez
#     schedule.Hv[t, u, v] = Ey
#
#     if t == 0 and u == 0 and v == 0:
#         schedule.OFFSETX = history.OFFSETX
#         schedule.OFFSETY = history.OFFSETY
#         schedule.OFFSETZ = history.OFFSETZ
#
#
# @wp.kernel
# def map_xz(
#     history: TEMStates,
#     schedule: TEMStates,
# ):
#     t, u, v = wp.tid()
#
#     Ex = history.Eu[t, u, v]
#     Ez = history.Ev[t, u, v]
#     Hx = history.Hu[t, u, v]
#     Hz = history.Hv[t, u, v]
#
#     schedule.Eu[t, u, v] = -Hz
#     schedule.Ev[t, u, v] = Hx
#     schedule.Hu[t, u, v] = Ez
#     schedule.Hv[t, u, v] = -Ex
#
#     if t == 0 and u == 0 and v == 0:
#         schedule.OFFSETX = history.OFFSETX
#         schedule.OFFSETY = history.OFFSETY
#         schedule.OFFSETZ = history.OFFSETZ
#
#
# @wp.kernel
# def map_xy(
#     history: TEMStates,
#     schedule: TEMStates,
# ):
#     t, u, v = wp.tid()
#
#     Ex = history.Eu[t, u, v]
#     Ey = history.Ev[t, u, v]
#     Hx = history.Hu[t, u, v]
#     Hy = history.Hv[t, u, v]
#
#     schedule.Eu[t, u, v] = Hy
#     schedule.Ev[t, u, v] = -Hx
#     schedule.Hu[t, u, v] = -Ey
#     schedule.Hv[t, u, v] = Ex
#
#     if t == 0 and u == 0 and v == 0:
#         schedule.OFFSETX = history.OFFSETX
#         schedule.OFFSETY = history.OFFSETY
#         schedule.OFFSETZ = history.OFFSETZ
#
#
# @wp.kernel
# def flip_h(source: TEMStates, target: TEMStates):
#     t, u, v = wp.tid()
#
#     target.Ev[t, u, v] = source.Ev[t, u, v]
#     target.Eu[t, u, v] = source.Eu[t, u, v]
#     target.Hv[t, u, v] = -source.Hv[t, u, v]
#     target.Hu[t, u, v] = -source.Hu[t, u, v]
#
#
# @wp.kernel
# def flip_axis_h(source: TEMStates, target: TEMStates):
#     t, u, v = wp.tid()
#
#     target.Ev[t, u, v] = source.Eu[t, v, u]
#     target.Eu[t, u, v] = source.Ev[t, v, u]
#     target.Hv[t, u, v] = -source.Hu[t, v, u]
#     target.Hu[t, u, v] = -source.Hv[t, v, u]
