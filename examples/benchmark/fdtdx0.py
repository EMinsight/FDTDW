import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx

c0 = fdtdx.constants.c
length_unit = 1e-6
time_unit = length_unit / c0
freq_unit = c0 / length_unit
resolution =1e-1 * length_unit
N=240

def main():
    exp_logger = fdtdx.Logger(
        experiment_name="simulate_source",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 1.55 * length_unit
    period = fdtdx.constants.wavelength_to_period(wavelength)

    object_list = []
    config = fdtdx.SimulationConfig(
        time=100 * time_unit,
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.5 * (3**0.5),
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

   
    constraints = []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(N*resolution,N*resolution,N*resolution),
        material=fdtdx.Material(  # Background material
            permittivity=1.0,
        ),
    )

    object_list.append(volume)
    periodic = False
    if periodic:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    else:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=20, boundary_type="pml"
        )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    object_list.extend(list(bound_dict.values()))
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(10 * length_unit, 10 * length_unit, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55 * length_unit),
        # radius=4*length_unit,
        # std=1 / 3,
        direction="-",
        elevation_angle=0.0,
    )

    object_list.append(source)
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 0),
            ),
        ]
    )



    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))



    def sim_fn(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key)

        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state



        new_info = {
            **info,
        }
        return arrays, new_info

    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = (
        jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    )
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_loss(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

if __name__ == "__main__":
    main()
