version = "0.2.0"
short_version = "0.2"

import sys
import importlib

def __get_mod_version(modname):
    """Get the version string of a module.
    
    Args:
        modname: Name of the module to check.
        
    Returns:
        Version string of the module, or None if not installed.
    """
    try:
        if modname in sys.modules:
            mod = sys.modules[modname]
        else:
            mod = importlib.import_module(modname)
        try:
            return mod.__version__
        except AttributeError:
            return "installed, no version number available"

    except ImportError:
        return None


def show_versions() -> None:
    """Print the version information for librosax and its dependencies.

    Prints a formatted list of all core and extra dependencies along with
    their installed versions.
    """
    core_deps = [
        "jax",
        "jaxlib",
        "flax",
        "einops",
        "librosa",
        "numpy",
        "scipy",
        "numba",
        "soundfile",
        "soxr",
        "lazy_loader",
        "typing_extensions",
    ]

    extra_deps = [
        "torch",
        "torchlibrosa",
        "matplotlib",
        "sphinx",
        "furo",
        "m2r2",
        "pytest",
        "pytest_cov",
    ]

    print("INSTALLED VERSIONS")
    print("------------------")
    print(f"python: {sys.version}\n")
    print(f"librosax: {version}\n")
    for dep in core_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))
    print("")
    for dep in extra_deps:
        print("{}: {}".format(dep, __get_mod_version(dep)))