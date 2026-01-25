from fdtdw import (
    ReferenceSimulation as Simulation,
    StandardMaterialModel,
    MaterialSimulation,
)
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

NX, NY, NZ = 64, 64, 1024
S = 0.56
sim = MaterialSimulation(
    Simulation(
        NX=NX,
        NY=NY,
        NZ=NZ,
        STEPS=2500,
        DEVICE="cuda:0",
        boundaries={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PML",
        },
        PML_THICKNESS=10,
        R_0=10e-12
    ),
    StandardMaterialModel(dx=0.01, S=S),
)

NU, NV = NX - 2, NY - 2

DSX = 1
DSY = 1
DSZ = NZ // 2

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

NUD = NX
NVD = NY
DDX = 0
DDY = 0

sim.init_detector(
    shape=(NUD, NVD), OFFSETX=DDX, OFFSETY=DDY, OFFSETZ=150 , plane="xy", full=True
)
sim.init_detector(
    shape=(NUD, NVD), OFFSETX=DDX, OFFSETY=DDY, OFFSETZ=NZ - 150 , plane="xy", full=True
)

sim.record_graphs()

sim.launch_forward()



for idx in (0, 1):

    flux=sim.detector_flux(idx=idx)

    flux_in=np.where(flux>0,flux,0)
    flux_out=np.where(flux<0,flux,0)
    p_in=np.trapezoid(np.abs(flux_in))
    p_out=np.trapezoid(np.abs(flux_out))
    r_p=p_out/p_in
    RL_dB=10*np.log10(r_p)
    print(f"RL_db:{RL_dB}")

    plt.plot(flux)

plt.xlabel("[n]")
plt.ylabel("$\int \int S dA$")
plt.savefig("flux.png")

sim.render_detector_video(field="Ev", filename="det.mp4", limit=1)
sim.render_source_video(field="Ev", filename="src.mp4", limit=1)
print(sim)
