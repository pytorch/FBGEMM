# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import jinja2

try:
    from .jinja_environment import env
    from .scripts_argsparse import args
except:
    # pyre-ignore[21]
    from jinja_environment import env

    # pyre-ignore[21]
    from scripts_argsparse import args


@dataclass
class CodeTemplate:
    template: jinja2.Template

    @staticmethod
    # pyre-ignore[3]
    def load(relative_path: str):
        return CodeTemplate(env.get_template(relative_path))

    def write(self, filename: str, **kwargs: Any) -> None:
        # Render the template
        output = self.template.render(**kwargs)

        # All generated files are written to the specified install directory.
        with open(os.path.join(args.install_dir, filename), "w") as f:
            f.write(output)
            print(f"Written: {filename}")

    @staticmethod
    def copy_to_root(relative_path: str) -> None:
        # Copy template from its relative path to root of the output directory
        # e.g. sub/directory/foo.py -> foo.py
        CodeTemplate.load(relative_path).write(relative_path.split("/")[-1])
