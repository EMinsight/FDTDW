from numpy import dtype
import warp as wp


@wp.struct
class EMState:
    Ex: wp.array3d(dtype=wp.float32)
    Ey: wp.array3d(dtype=wp.float32)
    Ez: wp.array3d(dtype=wp.float32)
    Hx: wp.array3d(dtype=wp.float32)
    Hy: wp.array3d(dtype=wp.float32)
    Hz: wp.array3d(dtype=wp.float32)

    psi_ex_y: wp.array3d(dtype=wp.float32)
    psi_ex_z: wp.array3d(dtype=wp.float32)
    psi_ey_x: wp.array3d(dtype=wp.float32)
    psi_ey_z: wp.array3d(dtype=wp.float32)
    psi_ez_x: wp.array3d(dtype=wp.float32)
    psi_ez_y: wp.array3d(dtype=wp.float32)

    psi_hx_y: wp.array3d(dtype=wp.float32)
    psi_hx_z: wp.array3d(dtype=wp.float32)
    psi_hy_x: wp.array3d(dtype=wp.float32)
    psi_hy_z: wp.array3d(dtype=wp.float32)
    psi_hz_x: wp.array3d(dtype=wp.float32)
    psi_hz_y: wp.array3d(dtype=wp.float32)


@wp.struct
class EMStates:
    """
    Shape: (CHECKPOINTS, NX, NY, NZ)
    """

    Ex: wp.array4d(dtype=wp.float32)
    Ey: wp.array4d(dtype=wp.float32)
    Ez: wp.array4d(dtype=wp.float32)
    Hx: wp.array4d(dtype=wp.float32)
    Hy: wp.array4d(dtype=wp.float32)
    Hz: wp.array4d(dtype=wp.float32)

    psi_ex_y: wp.array4d(dtype=wp.float32)
    psi_ex_z: wp.array4d(dtype=wp.float32)
    psi_ey_x: wp.array4d(dtype=wp.float32)
    psi_ey_z: wp.array4d(dtype=wp.float32)
    psi_ez_x: wp.array4d(dtype=wp.float32)
    psi_ez_y: wp.array4d(dtype=wp.float32)

    psi_hx_y: wp.array4d(dtype=wp.float32)
    psi_hx_z: wp.array4d(dtype=wp.float32)
    psi_hy_x: wp.array4d(dtype=wp.float32)
    psi_hy_z: wp.array4d(dtype=wp.float32)
    psi_hz_x: wp.array4d(dtype=wp.float32)
    psi_hz_y: wp.array4d(dtype=wp.float32)


@wp.struct
class DFTs:
    """
    Shape: (idx_freq, NX, NY, NZ)
    """

    Ex: wp.array4d(dtype=wp.vec2)
    Ey: wp.array4d(dtype=wp.vec2)
    Ez: wp.array4d(dtype=wp.vec2)
    Hx: wp.array4d(dtype=wp.vec2)
    Hy: wp.array4d(dtype=wp.vec2)
    Hz: wp.array4d(dtype=wp.vec2)


@wp.struct
class TEMDFTs:
    """
    Shape: (idx_freq, NU, NV)
    """

    Eu: wp.array3d(dtype=wp.vec2)
    Ev: wp.array3d(dtype=wp.vec2)
    Hu: wp.array3d(dtype=wp.vec2)
    Hv: wp.array3d(dtype=wp.vec2)


@wp.struct
class TEMDFT:
    """
    Shape: ( NU, NV)
    """

    Eu: wp.array2d(dtype=wp.vec2)
    Ev: wp.array2d(dtype=wp.vec2)
    Hu: wp.array2d(dtype=wp.vec2)
    Hv: wp.array2d(dtype=wp.vec2)


@wp.struct
class FieldsBuffer:
    """
    Shape: (BUFFERSIZE, NX, NY, NZ)
    """

    Ex: wp.array4d(dtype=wp.float32)
    Ey: wp.array4d(dtype=wp.float32)
    Ez: wp.array4d(dtype=wp.float32)
    Hx: wp.array4d(dtype=wp.float32)
    Hy: wp.array4d(dtype=wp.float32)
    Hz: wp.array4d(dtype=wp.float32)


@wp.struct
class TEMState:
    Eu: wp.array2d(dtype=wp.float32)
    Ev: wp.array2d(dtype=wp.float32)
    Hu: wp.array2d(dtype=wp.float32)
    Hv: wp.array2d(dtype=wp.float32)
    OFFSETX: wp.int32
    OFFSETY: wp.int32
    OFFSETZ: wp.int32


@wp.struct
class TEMStates:
    """
    Shape: (TIMESTEPS, NU, NV)
    """

    Eu: wp.array3d(dtype=wp.float32)
    Ev: wp.array3d(dtype=wp.float32)
    Hu: wp.array3d(dtype=wp.float32)
    Hv: wp.array3d(dtype=wp.float32)
    OFFSETX: wp.int32
    OFFSETY: wp.int32
    OFFSETZ: wp.int32


@wp.struct
class TEMStates_full:
    """
    Shape: (TIMESTEPS, NU, NV)
    """

    Eu: wp.array3d(dtype=wp.float32)
    Ev: wp.array3d(dtype=wp.float32)
    Hu: wp.array3d(dtype=wp.float32)
    Hv: wp.array3d(dtype=wp.float32)
    Hu_n: wp.array3d(dtype=wp.float32)
    Hv_n: wp.array3d(dtype=wp.float32)
    OFFSETX: wp.int32
    OFFSETY: wp.int32
    OFFSETZ: wp.int32


@wp.struct
class Gradients:
    grad_CE_x: wp.array3d(dtype=wp.float32)
    grad_CE_y: wp.array3d(dtype=wp.float32)
    grad_CE_z: wp.array3d(dtype=wp.float32)

    grad_CH_x: wp.array3d(dtype=wp.float32)
    grad_CH_y: wp.array3d(dtype=wp.float32)
    grad_CH_z: wp.array3d(dtype=wp.float32)


@wp.struct
class Properties:
    B_E_X: wp.array(dtype=wp.float32)
    C_E_X: wp.array(dtype=wp.float32)
    B_E_Y: wp.array(dtype=wp.float32)
    C_E_Y: wp.array(dtype=wp.float32)
    B_E_Z: wp.array(dtype=wp.float32)
    C_E_Z: wp.array(dtype=wp.float32)

    B_H_X: wp.array(dtype=wp.float32)
    C_H_X: wp.array(dtype=wp.float32)
    B_H_Y: wp.array(dtype=wp.float32)
    C_H_Y: wp.array(dtype=wp.float32)
    B_H_Z: wp.array(dtype=wp.float32)
    C_H_Z: wp.array(dtype=wp.float32)

    INV_K_X: wp.array(dtype=wp.float32)
    INV_K_Y: wp.array(dtype=wp.float32)
    INV_K_Z: wp.array(dtype=wp.float32)

    CEA_X: wp.array3d(dtype=wp.float32)
    CEA_Y: wp.array3d(dtype=wp.float32)
    CEA_Z: wp.array3d(dtype=wp.float32)
    CHA_X: wp.array3d(dtype=wp.float32)
    CHA_Y: wp.array3d(dtype=wp.float32)
    CHA_Z: wp.array3d(dtype=wp.float32)

    CEB_X: wp.array3d(dtype=wp.float32)
    CEB_Y: wp.array3d(dtype=wp.float32)
    CEB_Z: wp.array3d(dtype=wp.float32)
    CHB_X: wp.array3d(dtype=wp.float32)
    CHB_Y: wp.array3d(dtype=wp.float32)
    CHB_Z: wp.array3d(dtype=wp.float32)

    is_PML: wp.array3d(dtype=bool)
