import time
from typing import Union, TYPE_CHECKING

from fdtdw.materials._material_model import MaterialModel
from fdtdw.simulations._base_simulation import BaseSimulation

if TYPE_CHECKING:
    from fdtdw.simulations.adjoint_dft_simulation import AdjointDftSimulation
    from fdtdw.simulations.adjoint_cp_simulation import AdjointCpSimulation

    class CombinedSim(AdjointDftSimulation, AdjointCpSimulation):
        pass

    Parent = CombinedSim
else:
    Parent = object


class Style:
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


def pipe_logging(target_attr: str, methods: Union[list[str], dict[str, str]]):

    def decorator(cls):
        if isinstance(methods, list):
            method_map = {name: name for name in methods}
        else:
            method_map = methods

        for real_name, display_label in method_map.items():

            def wrapper(
                self, *args, _func_name=real_name, _label=display_label, **kwargs
            ):

                print(f"{Style.BOLD}{_label}{Style.RESET}...")
                t_start = time.perf_counter()

                inner_obj = getattr(self, target_attr)

                if not hasattr(inner_obj, _func_name):
                    raise AttributeError(
                        f"{Style.RED}The inner simulation ({type(inner_obj).__name__}) "
                        f"does not support the method '{_func_name}{Style.RESET}'."
                    )

                actual_func = getattr(inner_obj, _func_name)

                result = actual_func(*args, **kwargs)

                duration = time.perf_counter() - t_start
                print(f"... {Style.GREEN}done in {duration:.4f}s{Style.RESET}")

                return result

            setattr(cls, real_name, wrapper)
        return cls

    return decorator


def pipe_properties(target_attr: str, props: list[str]):
    def decorator(cls):
        for prop_name in props:

            def getter(self, name=prop_name):
                return getattr(getattr(self, target_attr), name)

            def setter(self, value, name=prop_name):
                setattr(getattr(self, target_attr), name, value)

            setattr(cls, prop_name, property(getter, setter))
        return cls

    return decorator


@pipe_logging(
    "sim",
    {
        "launch_forward": "Forward Simulation Run",
        "launch_adjoint": "Adjoint Simulation Run",
        "record_graphs": "CUDA Graph Recording",
        "sync": "GPU-CPU Synchronization",
        "init_source": "Source Initialization",
        "init_detector": "Detector Initialization",
        "generate_adjoint_source": "Adjoint Source Generation",
        "export_checkpoint": "Exporting Checkpoint Field (VTI)",
        "recompute_gradients": "Gradient Recomputation",
        "compute_pdf": "Pulse Density Function Calculation",
        "export_dft_to_vti": "Exporting DFT Fields (VTI)",
        "render_detector_video": "Rendering Detector Video",
        "render_source_video": "Rendering Source Video",
        "render_checkpoint_video": "Rendering Checkpoint Video",
        "plot_detektor_flux": "Plotting Detector Flux",
        "export_gradients": "Exporting Gradients (VTI)",
    },
)
@pipe_properties(
    "sim",
    [
        "state",
        "grads",
        "pec",
        "pmc",
        "detectors",
        "sources",
        "source_adj",
        "dft_weights",
        "DFT",
        "states",
    ],
)
class MaterialSimulation(Parent):
    def __init__(self, simulation: BaseSimulation, model: MaterialModel):
        self.sim = simulation
        self.model = model

        self.sim._dx = self.model.dx
        self.sim._S = self.model.S

        self.reset_to_vacuum()

    def reset_to_vacuum(self):
        shape = (self.sim._NX, self.sim._NY, self.sim._NZ)
        vacuum_params = self.model.get_vacuum_params(shape)
        coeffs = self.model.get_coeffs(vacuum_params)
        self._apply_coeffs(coeffs)

    def _apply_coeffs(self, coeffs: dict):
        for name, value in coeffs.items():
            setattr(self.sim, name, value)

    def _read_coeffs(self) -> dict:
        return {name: getattr(self.sim, name) for name in self.model.coeff_names}

    def set_parameters(self, params_dict: dict):
        coeffs = self.model.get_coeffs(params_dict)
        self._apply_coeffs(coeffs)

    def get_parameters(self) -> dict:
        current_coeffs = self._read_coeffs()
        return self.model.get_params(current_coeffs)

    def __getattr__(self, name):
        return getattr(self.sim, name)
