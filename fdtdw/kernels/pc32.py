import warp as wp
from .structs32 import Properties


@wp.kernel
def set_material_properties(
    val_CEB: wp.array3d(dtype=float),
    val_CEA: wp.array3d(dtype=float),
    val_CHB: wp.array3d(dtype=float),
    val_CHA: wp.array3d(dtype=float),
    FILL_PEC_X: wp.array3d(dtype=float),
    FILL_PEC_Y: wp.array3d(dtype=float),
    FILL_PEC_Z: wp.array3d(dtype=float),
    FILL_PMC_X: wp.array3d(dtype=float),
    FILL_PMC_Y: wp.array3d(dtype=float),
    FILL_PMC_Z: wp.array3d(dtype=float),
    props: Properties,
):
    i, j, k = wp.tid()

    safe_pec_x = 1.0 - FILL_PEC_X[i, j, k]
    props.CEB_X[i, j, k] = val_CEB[i, j, k] * safe_pec_x
    props.CEA_X[i, j, k] = val_CEA[i, j, k] * safe_pec_x

    safe_pec_y = 1.0 - FILL_PEC_Y[i, j, k]
    props.CEB_Y[i, j, k] = val_CEB[i, j, k] * safe_pec_y
    props.CEA_Y[i, j, k] = val_CEA[i, j, k] * safe_pec_y

    safe_pec_z = 1.0 - FILL_PEC_Z[i, j, k]
    props.CEB_Z[i, j, k] = val_CEB[i, j, k] * safe_pec_z
    props.CEA_Z[i, j, k] = val_CEA[i, j, k] * safe_pec_z

    safe_pmc_x = 1.0 - FILL_PMC_X[i, j, k]
    props.CHB_X[i, j, k] = val_CHB[i, j, k] * safe_pmc_x
    props.CHA_X[i, j, k] = val_CHA[i, j, k] * safe_pmc_x

    safe_pmc_y = 1.0 - FILL_PMC_Y[i, j, k]
    props.CHB_Y[i, j, k] = val_CHB[i, j, k] * safe_pmc_y
    props.CHA_Y[i, j, k] = val_CHA[i, j, k] * safe_pmc_y

    safe_pmc_z = 1.0 - FILL_PMC_Z[i, j, k]
    props.CHB_Z[i, j, k] = val_CHB[i, j, k] * safe_pmc_z
    props.CHA_Z[i, j, k] = val_CHA[i, j, k] * safe_pmc_z
