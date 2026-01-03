import warp as wp
from .structs32 import EMState, TEMStates


@wp.kernel
def inject_esources_yz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX
    gy = source.OFFSETY + u
    gz = source.OFFSETZ + v

    state.Ey[gx, gy, gz] += source.Eu[t, u, v]
    state.Ez[gx, gy, gz] += source.Ev[t, u, v]


@wp.kernel
def inject_esources_xz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY
    gz = source.OFFSETZ + v

    state.Ex[gx, gy, gz] += source.Eu[t, u, v]
    state.Ez[gx, gy, gz] += source.Ev[t, u, v]


@wp.kernel
def inject_esources_xy(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY + v
    gz = source.OFFSETZ

    state.Ex[gx, gy, gz] += source.Eu[t, u, v]
    state.Ey[gx, gy, gz] += source.Ev[t, u, v]


@wp.kernel
def substr_esources_yz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX
    gy = source.OFFSETY + u
    gz = source.OFFSETZ + v

    state.Ey[gx, gy, gz] -= source.Eu[t, u, v]
    state.Ez[gx, gy, gz] -= source.Ev[t, u, v]


@wp.kernel
def substr_esources_xz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY
    gz = source.OFFSETZ + v

    state.Ex[gx, gy, gz] -= source.Eu[t, u, v]
    state.Ez[gx, gy, gz] -= source.Ev[t, u, v]


@wp.kernel
def substr_esources_xy(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY + v
    gz = source.OFFSETZ

    state.Ex[gx, gy, gz] -= source.Eu[t, u, v]
    state.Ey[gx, gy, gz] -= source.Ev[t, u, v]


@wp.kernel
def inject_hsources_xz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY
    gz = source.OFFSETZ + v

    state.Hx[gx, gy, gz] += source.Hu[t, u, v]
    state.Hz[gx, gy, gz] += source.Hv[t, u, v]


@wp.kernel
def inject_hsources_yz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX
    gy = source.OFFSETY + u
    gz = source.OFFSETZ + v

    state.Hx[gx, gy, gz] += source.Hu[t, u, v]
    state.Hz[gx, gy, gz] += source.Hv[t, u, v]


@wp.kernel
def inject_hsources_xy(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY + v
    gz = source.OFFSETZ

    state.Hx[gx, gy, gz] += source.Hu[t, u, v]
    state.Hy[gx, gy, gz] += source.Hv[t, u, v]


@wp.kernel
def substr_hsources_yz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX
    gy = source.OFFSETY + u
    gz = source.OFFSETZ + v

    state.Hy[gx, gy, gz] -= source.Hu[t, u, v]
    state.Hz[gx, gy, gz] -= source.Hv[t, u, v]


@wp.kernel
def substr_hsources_xz(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY
    gz = source.OFFSETZ + v

    state.Hx[gx, gy, gz] -= source.Hu[t, u, v]
    state.Hz[gx, gy, gz] -= source.Hv[t, u, v]


@wp.kernel
def substr_hsources_xy(
    state: EMState, source: TEMStates, timestep_ptr: wp.array(dtype=wp.int32)
):
    u, v = wp.tid()
    t = timestep_ptr[0]

    gx = source.OFFSETX + u
    gy = source.OFFSETY + v
    gz = source.OFFSETZ

    state.Hx[gx, gy, gz] -= source.Hu[t, u, v]
    state.Hy[gx, gy, gz] -= source.Hv[t, u, v]
