#!/usr/bin/env bash
set -euo pipefail
NAME="$1"
CATEGORY="${2:-general}"
DIR="components/${NAME}"
mkdir -p "${DIR}/tests"
touch "${DIR}/__init__.py"
cat > "${DIR}/component.py" <<PYEOF
from pathlib import Path
import yaml
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_manifest = yaml.safe_load((Path(__file__).parent / "manifest.yaml").read_text())
_TARGET_IMAGE = f"ml-components/{_manifest['name']}:{_manifest['version']}"
_PACKAGE_NAME = "ml-components-$(echo "$NAME" | tr '_' '-')"

@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def ${NAME}() -> str:
    raise NotImplementedError("Implement ${NAME}")
PYEOF
cat > "${DIR}/manifest.yaml" <<YAMLEOF
name: $(echo "$NAME" | tr '_' '-')
version: "1.0.0"
category: ${CATEGORY}
description: |
  TODO: describe what this component does.
inputs: []
outputs:
  - name: output
    type: String
    description: TODO
    schema: "TODO"
typical_upstream:
  - TODO
typical_downstream:
  - TODO
resource_profile:
  cpu: "1"
  memory: "2Gi"
tags: [TODO]
YAMLEOF
cat > "${DIR}/tests/test_${NAME}.py" <<TESTEOF
def test_${NAME}_placeholder():
    pass
TESTEOF
echo "Scaffolded components/${NAME}/ — fill in component.py, manifest.yaml, and tests."
