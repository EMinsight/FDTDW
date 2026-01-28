import meep as mp

cell = mp.Vector3(12, 12, 12)

# geometry = [mp.Block(mp.Vector3(mp.inf,1,mp.inf),
#                      center=mp.Vector3(),
#                      material=mp.Medium(epsilon=12))]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=1 / 1.55),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0),
    )
]
pml_layers = [mp.PML(1.0)]
resolution = 10
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=[],
    sources=sources,
    resolution=resolution,
)

vals = []

# def get_slice(sim):
# vals.append(sim.get_array(center=mp.Vector3(0,-3.5), size=mp.Vector3(16,0), component=mp.Ez))

import time

start_time = time.time()
sim.run(
    # mp.at_beginning(mp.output_epsilon),
    # mp.at_every(6, get_slice),
    until=100
)
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.4f} seconds")

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(vals, interpolation='spline36', cmap='RdBu')
# plt.axis('off')
# print(len(vals))
# plt.show()
#
# plt.figure()
# ez_data = sim.get_array(center=mp.Vector3(0,0,0), size=(12,0,12), component=mp.Ez)
# plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
# plt.axis('off')
# plt.show()
