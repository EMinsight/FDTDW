from fdtdw import (
    AdjointCpSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

NX, NY, NZ = 128, 64, 224
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        STEPS=1500,
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
    StandardMaterialModel(dx=0.01, S=0.56),
)

NU, NV = 30, 10

DSX = (NX - NU) // 2
DSY = 1
DSZ = 20

width = 250.0
center = 600.0

OMEGA = 2.0 * np.pi / width

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


sim.init_detector(shape=(NU, NV), OFFSETX=DSX, OFFSETY=DSY, OFFSETZ=NZ - 20, plane="xy")

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

sim.render_checkpoint_video(
    slice_idx=(64, 5, 128), field="Ey", filename="strip.mp4", limit=1
)

sim.export_checkpoint_to_vti(filename="strip_fields.vti", pos=sim._CHECKPOINTS * 2 // 3)
print(sim)
