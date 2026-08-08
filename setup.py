"""Editable-install metadata for PIAD-V2X.

    pip install -e .

installs the pinned dependencies from requirements.txt and the `piad_v2x`
package, so the modules and the piad_v2x/tools/ entry points are importable anywhere.
"""
from pathlib import Path

from setuptools import find_packages, setup

_here = Path(__file__).parent
_req = _here / "requirements.txt"
install_requires = [
    line.strip()
    for line in _req.read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="piad-v2x",
    version="0.1.0",
    description=(
        "Physics-Informed Adaptive Defence for V2X: trust-gated pseudonym "
        "lifecycle and misbehaviour detection"
    ),
    author="Shaista Zeb",
    license="MIT",
    packages=find_packages(include=["piad_v2x", "piad_v2x.*"]),
    python_requires=">=3.11",
    install_requires=install_requires,
)
