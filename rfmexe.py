"""Path of the RFM executable of this plumeIR folder.

plumeIR ran './RFM/source/rfm', a relative path resolved by the process
launcher. On Windows that relies on CreateProcess adding '.exe' to a relative
path written with forward slashes. The absolute path, with '.exe' when that is
the file present (Windows builds), works on every platform. Results unchanged.
"""
import os

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RFM', 'source', 'rfm')


def rfm_executable():
    """RFM/source/rfm.exe when it exists and either this is Windows or there is
    no RFM/source/rfm, else RFM/source/rfm."""
    exe = _BASE + '.exe'
    if os.path.exists(exe) and (os.name == 'nt' or not os.path.exists(_BASE)):
        return exe
    return _BASE
