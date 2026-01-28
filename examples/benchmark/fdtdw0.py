from fdtdw import (
    ReferenceSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

sim = MaterialSimulation(
    Simulation(
        NX=240,
        NY=240,
        NZ=240,
        STEPS=2000,
        DEVICE="cuda:0",
        boundaries={
            "xmin": "PML",
            "xmax": "PML",
            "ymin": "PML",
            "ymax": "PML",
            "zmin": "PML",
            "zmax": "PML",
        },
        PML_THICKNESS=20,
    ),
    StandardMaterialModel(dx=0.01, S=0.50),
)

NX, NY, NZ = 120, 120, 120
NU, NV = 100, 100
DSX = (NX - NU) // 2
DSY = (NY - NV) // 2
DSZ = (NZ) // 2
width_t = 250.0
center_t = 600.0

OMEGA = 2.0 * np.pi / width_t

evfunc = lambda t: (1.0 - 2.0 * (np.pi * ((t - center_t) / width_t)) ** 2) * np.exp(
    -((np.pi * ((t - center_t) / width_t)) ** 2)
)
eufunc = lambda n: np.zeros_like(n)
euprofile = np.ones((NU, NV), dtype=np.float32)

evprofile = np.ones((NU, NV), dtype=np.float32)
hufunc = lambda n: np.zeros_like(n)
huprofile = np.ones((NU, NV), dtype=np.float32)
hvfunc = lambda n: np.zeros_like(n)

hvprofile = np.ones((NU, NV), dtype=np.float32)


sim.init_source(
    eufunc,
    euprofile,
    evfunc,
    evprofile,
    hufunc,
    huprofile,
    hvfunc,
    hvprofile,
    DSX,
    DSY,
    DSZ,
    "xy",
)


# sim.init_detector((NU, NV), DSX,DSY,DSZ+20, "xy")

sim.record_graphs()

sim.launch_forward()
# sim.render_detector_video(field="Ev", filename="det.mp4", limit=1)
sim.render_source_video(field="Ev", filename="src.mp4", limit=0.1)
print(sim)
