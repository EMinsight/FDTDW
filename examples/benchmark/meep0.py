import meep as mp
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=60)
parser.add_argument('--pml', type=int, default=1)
args = parser.parse_args()

N = args.N 
use_pml = bool(args.pml)

resolution = 10
cell = mp.Vector3(N/resolution, N/resolution, N/resolution)

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=1 / 1.55),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0),
    )
]

if use_pml: 
    pml_layers = [mp.PML(20 * 1.0 / resolution)] 
else:
    pml_layers = None

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=[],
    sources=sources,
    resolution=resolution,
)

sim.init_sim()

start_time = time.time()
sim.run(until=100)
end_time = time.time()

