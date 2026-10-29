"""Runtime setup shared by the command-line pipeline and dashboard."""

import os
import site
from pathlib import Path

_DLL_DIRECTORY_HANDLES = []


def configure_windows_dll_search():
    if os.name != "nt":
        return

    user_site = Path(site.getusersitepackages())
    if len(user_site.parents) < 2:
        return
    runtime_directory = user_site.parents[1]
    if (runtime_directory / "vcruntime140_threads.dll").is_file():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_directory)))
