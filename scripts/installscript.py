# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List

cli = argparse.ArgumentParser()
subparsers = cli.add_subparsers(dest="subcommand")

def argument(*name_or_flags, **kwargs):
    """Convenience function to properly format arguments to pass to the
    subcommand decorator.
    """
    return (list(name_or_flags), kwargs)

def subcommand(args=[], parent=subparsers):
    """Decorator to define a new subcommand in a sanity-preserving way.
    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so:

        args = cli.parse_args()
        args.func(args)

    Usage example:

        @subcommand([argument("-d", help="Enable debug mode", action="store_true")])
        def subcommand(args):
            print(args)

    Then on the command line:

        $ python cli.py subcommand -d
    """
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator

@dataclass
class InstallPyTorchCuda:
  '''Install PyTorch and CUDA packages'''
  conda_env: str
  version: str
  use_pip: bool

  def __exec_cmd(self) -> List[str]:
    conda = {
      'nightly' : 'conda install pytorch          pytorch-cuda=11.6 -c pytorch-nightly  -c nvidia',
      '1.13.1'  : 'conda install pytorch==1.13.1  pytorch-cuda=11.6 -c pytorch          -c nvidia',
      '1.12.1'  : 'conda install pytorch==1.12.1  cudatoolkit=11.6  -c pytorch          -c conda-forge',
      '1.12.0'  : 'conda install pytorch==1.12.0  cudatoolkit=11.6  -c pytorch          -c conda-forge',
      '1.11.0'  : 'conda install pytorch==1.11.0  cudatoolkit=11.3  -c pytorch',
    }

    pip = {
      'nightly' : 'pip3 install --pre torch               --extra-index-url https://download.pytorch.org/whl/nightly/cu116',
      '1.13.1'  : 'pip3 install       torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116',
      '1.12.1'  : 'pip3 install       torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116',
      '1.12.0'  : 'pip3 install       torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116',
      '1.11.0'  : 'pip3 install       torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113',
    }

    if self.use_pip and self.version in pip:
      return ['conda', 'run', '-n', self.conda_env] + pip[self.version].split()
    elif not self.use_pip and self.version in conda:
      return conda[self.version].split() + ['-n', self.conda_env, '-y']
    else:
      raise Exception(f"PyTorch version not supported for installation with the selected packge management system: {self.version}")

  def __check_cmd(self) -> List[str]:
    if self.use_pip:
      return f"conda run -n {self.conda_env} pip3 show torch".split()
    else:
      return f"conda list -n {self.conda_env} --json pytorch".split()

  def exec(self) -> None:
    command1 = self.__exec_cmd()
    print(f"Installing PyTorch: {command1}")
    subprocess.run(command1, check=True)

    command2 = self.__check_cmd()
    print(f"Verifying PyTorch installation: {command2}")

    if self.use_pip:
      subprocess.run(command2, check=True)

    else:
      proc = subprocess.run(command2, capture_output=True, check=True)
      packages = json.loads(str(proc.stdout, 'utf-8'))
      pytorches = [ p for p in packages if p['name'] == 'pytorch' ]
      cudnn = [ p for p in pytorches if 'cudnn' in p['dist_name'] ]
      if not pytorches:
        raise Exception("Failed to install PyTorch")
      elif not cudnn:
        raise Exception(f"PyTorch installation is not the GPU variant (i.e. does not contain CUDA / cuDNN)")

  def print(self) -> None:
    print('\n'.join([
      ' '.join(self.__exec_cmd()),
      ' '.join(self.__check_cmd()),
    ]))

@dataclass
class InstallBuildTools:
  '''Install build tools'''
  conda_env: str

  def __exec_cmd(self) -> List[str]:
    return f"conda install -n {self.conda_env} -y numpy scikit-build jinja2 cmake hypothesis".split()

  def exec(self) -> None:
    command = self.__exec_cmd()
    print(f"Installing build tools through Conda: {command}")
    subprocess.run(command, check=True)

  def print(self) -> None:
    print('\n'.join([
      ' '.join(self.__exec_cmd()),
    ]))

@dataclass
class InstallFBGEMM:
  '''Install FBGEMM package'''
  conda_env: str
  version: str

  def __exec_cmd(self) -> List[str]:
    if self.version == 'nightly':
      return f"conda run -n {self.conda_env} pip3 install fbgemm-gpu-nightly".split()
    elif self.version in ['0.1.1' '0.2.0', '0.3.0', '0.3.2']:
      return f"conda run -n {self.conda_env} pip3 install fbgemm-gpu=={self.version}".split()
    else:
      raise Exception(f"FBGEMM version not supported for installation: {self.version}")

  def exec(self) -> None:
    command = self.__exec_cmd()
    print(f"Installing build tools through Conda: {command}")
    subprocess.run(command, check=True)

  def print(self) -> None:
    print('\n'.join([
      ' '.join(self.__exec_cmd()),
    ]))

@dataclass
class FetchFBGEMMSource:
  '''Fetch FBGEMM source'''
  conda_env: str
  version: str
  directory: str

  tmpdir = tempfile.TemporaryDirectory()

  def __exec_cmd(self) -> List[str]:
    return f"git clone --recursive -b v{self.version} https://github.com/pytorch/FBGEMM.git {self.directory}".split()

  def exec(self) -> None:
    command = self.__exec_cmd()
    print(f"Fetching FBGEMM source: {command}")
    subprocess.run(command, check=True)

  def print(self) -> None:
    print('\n'.join([
      ' '.join(self.__exec_cmd()),
    ]))

# wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

@subcommand([
  argument("-x", "--execute",   help="Execute the command",         action="store_true"),
  argument("-n", "--conda-env", help="Conda Environment",           type=str, required=True),
  argument("-v", "--version",   help="PyTorch version",             type=str, default='nightly'),
  argument("-p", "--use-pip",   help="Use PIP to install PyTorch",  action="store_true"),
])
def install_pytorch(args):
  task = InstallPyTorchCuda(args.conda_env, args.version, args.use_pip)
  task.exec() if args.execute else task.print()

@subcommand([
  argument("-x", "--execute",   help="Execute the command",   action="store_true"),
  argument("-n", "--conda-env", help="Conda Environment",     type=str, required=True),
])
def install_build_tools(args):
  task = InstallBuildTools(args.conda_env)
  task.exec() if args.execute else task.print()

@subcommand([
  argument("-x", "--execute",   help="Execute the command",   action="store_true"),
  argument("-n", "--conda-env", help="Conda Environment",     type=str, required=True),
  argument("-v", "--version",   help="FBGEMM version",        type=str, default='0.2.0'),
])
def install_fbgemm(args):
  task = InstallFBGEMM(args.conda_env, args.version)
  task.exec() if args.execute else task.print()

if __name__ == "__main__":
  args = cli.parse_args()
  if args.subcommand is None:
      cli.print_help()
  else:
      args.func(args)
