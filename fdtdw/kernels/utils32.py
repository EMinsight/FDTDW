import warp as wp
from .structs32 import TEMStates_full, TEMStates


@wp.kernel
def calc_flux_yz(detector: TEMStates_full, flux_history: wp.array(dtype=wp.float32)):

    t, u, v = wp.tid()

    eu = detector.Eu[t, u, v]
    ev = detector.Ev[t, u, v]
    if t > 0:
        hu = 0.25 * (
            detector.Hu[t, u, v]
            + detector.Hu[t - 1, u, v]
            + detector.Hu_n[t, u, v]
            + detector.Hu_n[t - 1, u, v]
        )
        hv = 0.25 * (
            detector.Hv[t, u, v]
            + detector.Hv[t - 1, u, v]
            + detector.Hv_n[t, u, v]
            + detector.Hv_n[t - 1, u, v]
        )
    else:
        hu = 0.0
        hv = 0.0

    val = eu * hv - ev * hu

    wp.atomic_add(flux_history, t, val)


@wp.kernel
def calc_flux_xz(detector: TEMStates_full, flux_history: wp.array(dtype=wp.float32)):

    t, u, v = wp.tid()

    eu = detector.Eu[t, u, v]
    ev = detector.Ev[t, u, v]
    hu = detector.Hu[t, u, v]
    hv = detector.Hv[t, u, v]
    if t > 0:
        hu = 0.25 * (
            detector.Hu[t, u, v]
            + detector.Hu[t - 1, u, v]
            + detector.Hu_n[t, u, v]
            + detector.Hu_n[t - 1, u, v]
        )
        hv = 0.25 * (
            detector.Hv[t, u, v]
            + detector.Hv[t - 1, u, v]
            + detector.Hv_n[t, u, v]
            + detector.Hv_n[t - 1, u, v]
        )
    else:
        hu = 0.0
        hv = 0.0

    val = ev * hu - eu * hv

    wp.atomic_add(flux_history, t, val)


@wp.kernel
def calc_flux_xy(detector: TEMStates_full, flux_history: wp.array(dtype=wp.float32)):

    t, u, v = wp.tid()

    eu = detector.Eu[t, u, v]
    ev = detector.Ev[t, u, v]
    hu = detector.Hu[t, u, v]
    hv = detector.Hv[t, u, v]
    if t > 0:
        hu = 0.25 * (
            detector.Hu[t, u, v]
            + detector.Hu[t - 1, u, v]
            + detector.Hu_n[t, u, v]
            + detector.Hu_n[t - 1, u, v]
        )
        hv = 0.25 * (
            detector.Hv[t, u, v]
            + detector.Hv[t - 1, u, v]
            + detector.Hv_n[t, u, v]
            + detector.Hv_n[t - 1, u, v]
        )
    else:
        hu = 0.0
        hv = 0.0

    val = eu * hv - ev * hu

    wp.atomic_add(flux_history, t, val)


@wp.kernel
def inc_integer(ptr: wp.array(dtype=int), val: int):
    ptr[0] += val


@wp.kernel
def dec_integer(ptr: wp.array(dtype=int), val: int):
    ptr[0] -= val


@wp.kernel
def reset_integer(ptr: wp.array(dtype=int)):
    ptr[0] = 0
