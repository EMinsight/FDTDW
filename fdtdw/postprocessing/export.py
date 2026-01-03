import warp as wp
import numpy as np
import pyvista as pv
from typing import Union, Tuple, Any


def to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.array(data)


def export_vti(
    filename: str,
    fields: dict[str, Union[Tuple, np.ndarray, Any]],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    roi: Any = np.s_[:, :, :],
) -> None:

    if not fields:
        return

    first_key = next(iter(fields))
    first_val = fields[first_key]

    if isinstance(first_val, (tuple, list)):
        ref_raw = first_val[0]
    else:
        ref_raw = first_val

    ref_arr = to_numpy(ref_raw)[roi]
    shape = ref_arr.shape

    grid = pv.ImageData()
    grid.dimensions = shape
    grid.spacing = spacing

    for name, data in fields.items():
        if isinstance(data, (tuple, list)):
            if len(data) != 3:
                continue

            c1 = to_numpy(data[0])[roi]
            c2 = to_numpy(data[1])[roi]
            c3 = to_numpy(data[2])[roi]

            vec = np.stack((c1, c2, c3), axis=-1)

            if c1.shape != shape:
                continue

            grid.point_data[name] = vec.reshape(-1, 3, order="F")
            grid.point_data[f"Mag_{name}"] = np.linalg.norm(vec, axis=-1).flatten(
                order="F"
            )

        else:
            scalar_arr = to_numpy(data)[roi]

            if scalar_arr.shape != shape:
                continue

            grid.point_data[name] = scalar_arr.flatten(order="F")

    grid.save(filename)
    print(f"Saved VTI to {filename}")
