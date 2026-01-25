from fdtdw import (
    AdjointCpSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

XW = 100
NX, NY, NZ = 138 + XW, 64, 138
S = 0.56
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        STEPS=3000,
        DEVICE="cuda:0",
        boundaries={
            "xmin": "PML",
            "xmax": "PML",
            "ymin": "PEC",
            "ymax": "PML",
            "zmin": "PML",
            "zmax": "PML",
        },
        PML_THICKNESS=10,
    ),
    StandardMaterialModel(dx=0.01, S=S),
)

width = 400.0
center =1000.0
OMEGA = 2.0 * np.pi / width
OMEGA_t = OMEGA / S
target_file = f"mode_profile_w{OMEGA_t:.4f}.npz"
if not os.path.exists(target_file):
    files = glob.glob("mode_profile_w*.npz")
    if files:
        target_file = files[0]
        print(f"frequency not found. Using: {target_file}")
    else:
        raise FileNotFoundError("No profile found. Run mode_rec.py first.")

with np.load(target_file) as data:
    Eu_mode_in = data["Eu"]
    Ev_mode_in = data["Ev"]
    Hu_mode_in = data["Hu"]
    Hv_mode_in = data["Hv"]

W_CAPT, H_CAPT = 88, 44
OFFSET_SRC_X = (NX - W_CAPT + XW) // 2
OFFSET_SRC_Y = 1
OFFSET_SRC_Z = 15

time_func = lambda t: (1.0 - 2.0 * (np.pi * ((t - center) / width)) ** 2) * np.exp(
    -((np.pi * ((t - center) / width)) ** 2)
)
zero_func = lambda n: np.zeros_like(n)

sim.init_source(
    eufunc=time_func,
    euprofile=Eu_mode_in,
    evfunc=time_func,
    evprofile=Ev_mode_in,
    hufunc=time_func,
    huprofile=Hu_mode_in,
    hvfunc=time_func,
    hvprofile=Hv_mode_in,
    OFFSETX=OFFSET_SRC_X,
    OFFSETY=OFFSET_SRC_Y,
    OFFSETZ=OFFSET_SRC_Z,
    plane="xy",
)

OFFSET_DET_X = 15
OFFSET_DET_Y = 1
OFFSET_DET_Z = (NZ - W_CAPT) // 2

sim.init_detector(
    shape=(H_CAPT, W_CAPT),
    OFFSETX=OFFSET_DET_X,
    OFFSETY=OFFSET_DET_Y,
    OFFSETZ=OFFSET_DET_Z,
    plane="yz",
)

region_size = 10
val_background = 1.0
val_region = 4.4
sigma_region = 0.027

eps_r = np.ones((NX, NY, NZ), dtype=np.float32) * val_background
sigma_e = np.zeros((NX, NY, NZ), dtype=np.float32)

start_y = 0
end_y = start_y + region_size
eps_r[:, start_y:end_y, :] = val_region
sigma_e[:, start_y:end_y, :] = sigma_region

mdir = sim.get_parameters()
mdir["eps_r"] = eps_r
mdir["sigma_e"] = sigma_e
sim.set_parameters(mdir)

W_STRIP = 30
H_STRIP = 10

strip_start_x = (NX - W_STRIP + XW) // 2
strip_start_z = (NZ - W_STRIP) // 2

pec_map = np.zeros((NX, NY, NZ))

pec_map[
    strip_start_x : strip_start_x + W_STRIP + 1,
    H_STRIP,
    0 : strip_start_z + W_STRIP + 1,
] = 1

pec_map[0:strip_start_x, H_STRIP, strip_start_z : strip_start_z + W_STRIP + 1] = 1

x_outer_corner = strip_start_x + W_STRIP
z_outer_corner = strip_start_z + W_STRIP
miter_size = 20 * np.sqrt(2)

for x in range(strip_start_z, x_outer_corner + 1):
    for z in range(strip_start_z, z_outer_corner + 1):
        if x + z > x_outer_corner + z_outer_corner - miter_size:
            pec_map[x, H_STRIP, z] = 0

sim.pec = pec_map

sim.record_graphs()
sim.launch_forward()


adjoint_data_list = []

for det in sim.detectors:
    Eu_rec = det["Eu"]
    Ev_rec = det["Ev"]
    Hu_rec = det["Hu"]
    Hv_rec = det["Hv"]

    for H_rec in (Hu_rec, Hv_rec):
        H_prev = np.pad(H_rec, ((1, 0), (0, 0), (0, 0)))[:-1]
        H_aligned = 0.5 * (H_prev + H_rec)
        H_rec[:] = H_aligned

    Eu_T = Eu_mode_in.T
    Ev_T = Ev_mode_in.T
    Hu_T = Hu_mode_in.T
    Hv_T = Hv_mode_in.T

    Eu_mode = Ev_T
    Ev_mode = Eu_T
    Hu_mode = -1.0 * Hv_T
    Hv_mode = -1.0 * Hu_T

    omega_adj = OMEGA_t
    dt = 0.56
    phase_correction = np.exp(-1j * omega_adj * dt / 2)

    Hu_mode = Hu_mode * phase_correction
    Hv_mode = Hv_mode * phase_correction

    term1 = (Eu_rec * np.conj(Hv_mode)) - (Ev_rec * np.conj(Hu_mode))
    term2 = (np.conj(Eu_mode) * Hv_rec) - (np.conj(Ev_mode) * Hu_rec)

    integrand = term1 + term2
    du = dv = 1
    alpha_t = np.sum(integrand, axis=(-2, -1)) * (du * dv)

    weight_t = np.conj(alpha_t)

    plt.figure()
    plt.plot(np.abs(weight_t) / 10e5)
    plt.xlabel("n")
    plt.ylabel("|mode overlap|")
    plt.savefig("overlap.png")

    weight_t_broadcast = weight_t[:, None, None]

    Eu_mode_b = Eu_mode[None, :, :]
    Ev_mode_b = Ev_mode[None, :, :]
    Hu_mode_b = Hu_mode[None, :, :]
    Hv_mode_b = Hv_mode[None, :, :]

    adj_dict = {
        "Eu": np.real(Eu_mode_b * weight_t_broadcast).astype(np.float32),
        "Ev": np.real(Ev_mode_b * weight_t_broadcast).astype(np.float32),
        "Hu": -1.0 * np.real(Hu_mode_b * weight_t_broadcast).astype(np.float32),
        "Hv": -1.0 * np.real(Hv_mode_b * weight_t_broadcast).astype(np.float32),
        "OFFSETX": det["OFFSETX"],
        "OFFSETY": det["OFFSETY"],
        "OFFSETZ": det["OFFSETZ"],
    }
    adjoint_data_list.append(adj_dict)

sim.source_adj = adjoint_data_list

sim.launch_adjoint()

sim.export_gradients(filename="mode_corner_grads.vti", roi=np.s_[50:200, :, :])

sim.export_checkpoint_to_vti(filename="corner_fields.vti", pos=sim._CHECKPOINTS // 2)
sim.render_checkpoint_video(
    slice_idx=(NX // 2, 5, NZ // 2), field="Ey", filename="mtest.mp4", limit=300
)
sim.render_detector_video(field="Ev", filename="det_c.mp4", limit=10)
