import subprocess
import sys
import os

LATEST_AML_COMMIT = "27f518c7247dfb78d3de4d4066ef31f28921d46c"


def install_aml_storage(commit="latest"):
    if commit == "latest":
        commit = LATEST_AML_COMMIT

    try:
        import aml_storage

        package_root = os.path.dirname(aml_storage.__file__)
        if os.path.isfile(os.path.join(package_root, commit)):
            print(f"aml_storage @ {commit} is already installed")
            return

    except ImportError:
        pass

    package = (
        f"aml_storage @ https://github.com/Luthaf/aml-storage/archive/{commit}.zip"
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    import aml_storage

    package_root = os.path.dirname(aml_storage.__file__)
    with open(os.path.join(package_root, commit), "w"):
        pass
