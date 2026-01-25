from fdtdw import (
    ReferenceSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

NX, NY, NZ = 128, 64, 512
S = 0.56
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        STEPS=4000,
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

NU, NV = 30, 10

DSX = (NX - NU) // 2
DSY = 1
DSZ = 20

width = 250.0
center = 600.0

OMEGA = 2.0 * np.pi / width
OMEGA_t = 2.0 * np.pi / width / S

evfunc = lambda t: (1.0 - 2.0 * (np.pi * ((t - center) / width)) ** 2) * np.exp(
    -((np.pi * ((t - center) / width)) ** 2)
)
eufunc = lambda n: np.zeros_like(n)
euprofile = np.ones((NU, NV), dtype=np.float32)

evprofile = np.ones((NU, NV), dtype=np.float32)
hufunc = lambda n: np.zeros_like(n)
huprofile = np.ones((NU, NV), dtype=np.float32)
hvfunc = lambda n: np.zeros_like(n)

hvprofile = np.ones((NU, NV), dtype=np.float32)

sim.init_source(
    eufunc=eufunc,
    euprofile=euprofile,
    evfunc=evfunc,
    evprofile=evprofile,
    hufunc=hufunc,
    huprofile=huprofile,
    hvfunc=hvfunc,
    hvprofile=hvprofile,
    OFFSETX=DSX,
    OFFSETY=DSY,
    OFFSETZ=DSZ,
    plane="xy",
)

NUD = NX - 2 * 20
NVD = NY - 20
DDX = (NX - NUD) // 2
DDY = 1
DDZ1 = DSZ + 5
DDZ2 = NZ - 20
sim.init_detector(
    shape=(NUD, NVD), OFFSETX=DDX, OFFSETY=DDY, OFFSETZ=DDZ1, plane="xy", full=True
)
sim.init_detector(
    shape=(NUD, NVD), OFFSETX=DDX, OFFSETY=DDY, OFFSETZ=DDZ2, plane="xy", full=True
)
sub_height = 10
eps_r_air = 1
eps_r_sub = 4.4

eps_r = np.ones((NX, NY, NZ), dtype=np.float32) * eps_r_air
start = 0
end = start + sub_height

eps_r[:, start:end, :] = eps_r_sub

sigma_e = np.zeros((NX, NY, NZ), dtype=np.float32)
start = 0
end = start + sub_height
sigma_e[:, start:end, :] = sigma_e_sub = 0.027

mdir = sim.get_parameters()
mdir["eps_r"] = eps_r
mdir["sigma_e"] = sigma_e
sim.set_parameters(mdir)

arr1 = np.zeros((NX, NY, NZ))

arr1[DSX : DSX + NU + 1, NV, :] = 1
sim.pec = arr1

sim.record_graphs()

sim.launch_forward()


target_freqs = np.array([OMEGA_t])
dft_detector_1 = sim.get_dft_detector(target_freqs, idx=0, save=False, collocated=True)
dft_detector_2 = sim.get_dft_detector(target_freqs, idx=1, save=True, collocated=True)

fields = ["Ev", "Eu", "Hu", "Hv"]
ND = 0
for dft_detector in (dft_detector_1, dft_detector_2):
    ND += 1
    for field in fields:

        raw_data = dft_detector[field][0]

        complex_field = raw_data[..., 0] + 1j * raw_data[..., 1]

        magnitude = np.abs(complex_field)
        phase = np.angle(complex_field)

        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        im_mag = ax[0].imshow(
            magnitude.T, origin="lower", cmap="inferno", aspect="equal"
        )

        ax[0].set_title(f"|{field}|\n$\omega$={OMEGA_t:.3f}")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        plt.colorbar(im_mag, ax=ax[0], label="Field Strength")

        im_phase = ax[1].imshow(
            phase.T,
            origin="lower",
            cmap="twilight",
            aspect="equal",
            vmin=-np.pi,
            vmax=np.pi,
        )

        ax[1].set_title(f"Arg({field})\n$\omega$={OMEGA_t:.3f}")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        cbar = plt.colorbar(
            im_phase, ax=ax[1], label="Radians", ticks=[-np.pi, 0, np.pi]
        )
        cbar.ax.set_yticklabels(["$-\pi$", "$0$", "$\pi$"])

        plt.tight_layout()
        filename = f"profile_{field}_{ND}_coll.png"
        plt.savefig(filename)
        plt.close(fig)



sim.render_detector_video(field="Ev", filename="det.mp4", limit=1)
sim.render_source_video(field="Ev", filename="src.mp4", limit=1)
print(sim)
