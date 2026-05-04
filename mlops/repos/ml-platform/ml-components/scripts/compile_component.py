#!/usr/bin/env python3
"""Compile a KFP component directory to component.yaml.

Usage:
    python scripts/compile_component.py <component_name>

Example:
    python scripts/compile_component.py model_training
    # Generates: components/model_training/component.yaml
"""

import sys
import importlib
import pathlib

sys.path.insert(0, ".")

comp = sys.argv[1]
pkg_name = f"ml_components_{comp}"
mod = importlib.import_module(f"components.{comp}.{pkg_name}.component")
fn = getattr(mod, comp)

from kfp import compiler

out = pathlib.Path(f"components/{comp}/component.yaml")
compiler.Compiler().compile(fn, str(out))
print(f"Generated {out}")
