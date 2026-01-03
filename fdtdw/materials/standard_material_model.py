import numpy as np
from typing import List
from fdtdw.materials._material_model import MaterialModel


class StandardMaterialModel(MaterialModel):
    def __init__(self, dx: float, S: float):

        self.c = 1.0
        self.mu_0 = 1.0
        self.eps_0 = 1.0
        self.eta_0 = 1.0

        super().__init__(dx=dx, S=S)
        self.dt = (self.S * self.dx) / self.c

    @property
    def coeff_names(self) -> List[str]:
        return ["cea", "ceb", "cha", "chb"]

    def get_vacuum_params(self, shape: tuple) -> dict:
        return {
            "eps_r": np.ones(shape, dtype=np.float32),
            "mu_r": np.ones(shape, dtype=np.float32),
            "sigma_e": np.zeros(shape, dtype=np.float32),
            "sigma_m": np.zeros(shape, dtype=np.float32),
        }

    def get_coeffs(self, params: dict) -> dict:
        eps_r = params["eps_r"]
        mu_r = params["mu_r"]
        sigma_e = params["sigma_e"]
        sigma_m = params["sigma_m"]
        eps = eps_r * self.eps_0
        mu = mu_r * self.mu_0

        denom_e = 2.0 * eps + sigma_e * self.dt
        cea = (2.0 * eps - sigma_e * self.dt) / denom_e
        ceb = (2.0 * self.dt) / (denom_e * self.dx)

        denom_h = 2.0 * mu + sigma_m * self.dt
        cha = (2.0 * mu - sigma_m * self.dt) / denom_h
        chb = (2.0 * self.dt) / (denom_h * self.dx)

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

        D_e = (2.0 * self.dt) / (ceb * self.dx)
        eps_abs = (D_e * (1.0 + cea)) * 0.25
        sigma_e = (D_e * (1.0 - cea)) / (2.0 * self.dt)

        D_h = (2.0 * self.dt) / (chb * self.dx)
        mu_abs = (D_h * (1.0 + cha)) * 0.25
        sigma_m = (D_h * (1.0 - cha)) / (2.0 * self.dt)

        return {
            "eps_r": (eps_abs / self.eps_0).astype(np.float32),
            "mu_r": (mu_abs / self.mu_0).astype(np.float32),
            "sigma_e": sigma_e.astype(np.float32),
            "sigma_m": sigma_m.astype(np.float32),
        }
