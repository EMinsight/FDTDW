import time
import csv
import jax
import jax.numpy as jnp
import pytreeclass as tc
# from loguru import logger
import fdtdx

c0 = fdtdx.constants.c
length_unit = 1e-6
time_unit = length_unit / c0
resolution = 1e-1 * length_unit

def run(N, is_periodic):

    

    
    key = jax.random.PRNGKey(seed=42)
    wavelength = 1.55 * length_unit
    
    config = fdtdx.SimulationConfig(
        time=100 * time_unit,
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.5 * (3**0.5),
    )

    object_list = []
    
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(N * resolution, N * resolution, N * resolution),
        material=fdtdx.Material(permittivity=1.0),
    )
    object_list.append(volume)

    constraints = []

    if is_periodic:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    else:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=20, boundary_type="pml"
        )

    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    source_size = 20 * resolution
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(source_size, source_size, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55 * length_unit),
        direction="-",
        elevation_angle=0.0,
    )
    object_list.append(source)
    constraints.extend([
        source.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0)
        ),
    ])

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )

    def sim_fn(params, arrays, key):
        arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key)
        final_state = fdtdx.run_fdtd(arrays=arrays, objects=new_objects, config=config, key=key)
        _, arrays = final_state
        return arrays

    jitted_sim = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    
    key, subkey = jax.random.split(key)
    run_start_time = time.time()
    
    arrays = jitted_sim(params, arrays, subkey)
    
    
    runtime_delta = time.time() - run_start_time
    
    del jitted_sim, arrays, objects, params
    jax.clear_caches() 
    
    return runtime_delta

if __name__ == "__main__":
    sizes = list(range(40, 129, 8)) + list(range(160, 449, 32))
    
    modes = [
        (False, "fdtdx_pml"), 
        (True, "fdtdx_periodic")
    ]
    
    output_file = "results_fdtdx.csv"
    
    print(f"Starting Benchmark. Saving to {output_file}...")
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kernel", "N", "time"])
        f.flush()         
        for N in sizes:
            for is_periodic, kernel_name in modes:
                try:
                    print(f"Running {kernel_name} | N={N} ... ", end="", flush=True)
                    
                    runtime = run(N, is_periodic)
                    
                    writer.writerow([kernel_name, N, runtime])
                    f.flush() 
                    
                    print(f"{runtime:.4f} s")
                    
                except Exception as e:
                    print(f"FAILED! Error: {e}")
                    
    print("\nBenchmark Complete.")
