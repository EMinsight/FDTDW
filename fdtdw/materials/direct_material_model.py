import numpy as np
from typing import List
from fdtdw.materials._material_model import MaterialModel


class DirektMaterialModel(MaterialModel):
    def __init__(self, dx: float, S: float):

        super().__init__(dx=dx, S=S)

    @property
    def coeff_names(self) -> List[str]:
        return ["cea", "ceb", "cha", "chb"]

    def get_vacuum_params(self, shape: tuple) -> dict:
        return {
            "cea": np.ones(shape, dtype=np.float32),
            "ceb": np.ones(shape, dtype=np.float32) * self.S,
            "cha": np.ones(shape, dtype=np.float32),
            "chb": np.ones(shape, dtype=np.float32) * self.S,
        }

    def get_coeffs(self, params: dict) -> dict:
        cea = params["cea"]
        ceb = params["ceb"]
        cha = params["cha"]
        chb = params["chb"]

        return {
            "cea": cea.astype(np.float32),
            "ceb": ceb.astype(np.float32),
            "cha": cha.astype(np.float32),
            "chb": chb.astype(np.float32),
        }

    def get_params(self, coeffs: dict) -> dict:
        cea = np.array(coeffs["cea"], dtype=np.float32)
        ceb = np.array(coeffs["ceb"], dtype=np.float32)
        cha = np.array(coeffs["cha"], dtype=np.float32)
        chb = np.array(coeffs["chb"], dtype=np.float32)

        return {"cea": cea, "ceb": ceb, "cha": cha, "chb": chb}
