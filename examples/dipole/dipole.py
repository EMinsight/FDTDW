from fdtdw import (
    AdjointCpSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

NX, NY, NZ = 128, 128, 128
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        CHECKPOINTS=50,
        BUFFERSIZE=4,
        DEVICE="cuda:0",
        boundaries={
            "xmin": "PML",
            "xmax": "PML",
            "ymin": "PML",
            "ymax": "PML",
            "zmin": "PML",
            "zmax": "PML",
        },
        PML_THICKNESS=10,
    ),
    StandardMaterialModel(dx=0.01, S=0.56),
)

NU, NV = 10, 10
OMEGA = 2.0 * np.pi / 60.0

DSX = (NX - NU) // 2
DSY = (NY - NV) // 2
DSZ = NZ // 2

# width_t = 250.0
# center_t = 600.0
#
# OMEGA = 2.0 * np.pi / width_t

evfunc = lambda t: np.sin(t * OMEGA)
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

NU, NV = 108, 108

DSX = (NX - NU) // 2
DSY = (NY - NV) // 2
DSZ = NZ // 2
sim.init_detector(shape=(NU, NV), OFFSETX=DSX, OFFSETY=DSY, OFFSETZ=NZ - 15, plane="xy")

# region_size = 10
# val_background = 1
# val_region = 4.4
#
# arr = np.ones((NX, NY, NZ), dtype=np.float32) * val_background
# start = 0
# end = start + region_size
#
# arr[:, start:end, :] = val_region
#
# arrs = np.zeros((NX, NY, NZ), dtype=np.float32)
# start = 0
# end = start + region_size
# arrs[:, start:end, :] = 0.027
#
# mdir = sim.get_parameters()
# mdir["eps_r"] = arr
# mdir["sigma_e"] = arrs
# sim.set_parameters(mdir)
#
# arr1 = np.zeros((NX, NY, NZ))
# arr1[:, 0, :] = 1
#
# arr1[DSX : DSX + NU + 1, NV, :] = 1
# sim.pec = arr1


sim.record_graphs()

sim.launch_forward()
# sim.generate_adjoint_source()
# sim.launch_adjoint()

sim.render_checkpoint_video(
    slice_idx=(NX // 2, NY // 2, NZ // 2), field="Ey", filename="dipole.mp4", limit=0.3
)
sim.render_detector_video(field="Ev", filename="dipole_det.mp4", limit=0.3)
sim.render_source_video(field="Ev", filename="dipole_src.mp4", limit=0.3)

sim.export_checkpoint_to_vti(filename="dipole_fields.vti", pos=sim._CHECKPOINTS - 1)
# sim.export_gradients()
print(sim)
