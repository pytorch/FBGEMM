#################################################################################################
#
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################
# @lint-ignore-every LICENSELINT

import logging
import os
import re
import shutil

import cutlass.python.cutlass_library.library as library

from .emitters import EmitOperationKindLibrary
from .gemm_operation import EmitGemmOperationInstance

# fmt: off
###################################################################################################
_LOGGER = logging.getLogger(__name__)

class Manifest:

  def __init__(self, args = None):
    self.operations = {}
    self.args = args
    self.operation_count = 0
    self.operations_by_name = {}

    self.kernel_filter = ''
    self.kernel_filter_list = []
    self.kernel_names = []
    self.operations_enabled = []
    self.selected_kernels = []
    self.ignore_kernel_names = []
    self.compute_capabilities = [50,]
    self.curr_build_dir = '.'
    self.filter_by_cc = True

    if self.args:
      self.kernel_filter = self.args.kernels
      self.curr_build_dir = args.curr_build_dir

      # A common user error is to use commas instead of semicolons.
      if ',' in args.architectures:
        raise RuntimeError("The list of architectures (CMake option CUTLASS_NVCC_ARCHS) must be semicolon-delimited.\nDon't use commas to separate the architectures; use semicolons.\nYou specified the list as: " + args.architectures)
      architectures = args.architectures.split(';') if len(args.architectures) else ['50',]

      arch_conditional_cc = ['90a']
      architectures = [x if x not in arch_conditional_cc else x.split('a')[0] for x in architectures]

      self.compute_capabilities = [int(x) for x in architectures]

      if args.filter_by_cc in ['false', 'False', '0']:
        self.filter_by_cc = False

    if args.operations == 'all':
      self.operations_enabled = []
    else:
      operations_list = [
        library.OperationKind.Gemm,
      ]
      self.operations_enabled = [x for x in operations_list if library.OperationKindNames[x] in args.operations.split(',')]

    if args.kernels == 'all':
      self.kernel_names = []
    else:
      self.kernel_names = [x for x in args.kernels.split(',') if x != '']

    self.ignore_kernel_names = [x for x in args.ignore_kernels.split(',') if x != '']

    if args.kernel_filter_file is None:
        self.kernel_filter_list = []
    else:
        self.kernel_filter_list = self.get_kernel_filters(args.kernel_filter_file)
        _LOGGER.debug("Using {filter_count} kernel filters from {filter_file}".format(
            filter_count = len(self.kernel_filter_list),
            filter_file = args.kernel_filter_file))

    self.operation_count = 0
    self.operations_by_name = {}

  def get_kernel_filters(self, kernelListFile):
    if os.path.isfile(kernelListFile):
        with open(kernelListFile, 'r') as fileReader:
            lines = [line.rstrip() for line in fileReader if not line.startswith("#")]

        lines = [re.compile(line) for line in lines if line]
        return lines
    else:
        return []

  def filter_out_kernels(self, kernel_name, kernel_filter_list):

    for kernel_filter_re in kernel_filter_list:
        if kernel_filter_re.search(kernel_name) is not None:
            return True

    return False


  def _filter_string_matches(self, filter_string, haystack):
    ''' Returns true if all substrings appear in the haystack in order'''
    substrings = filter_string.split('*')
    for sub in substrings:
      idx = haystack.find(sub)
      if idx < 0:
        return False
      haystack = haystack[idx + len(sub):]
    return True

  def filter(self, operation):
    ''' Filtering operations based on various criteria'''

    # filter based on compute capability
    enabled = not (self.filter_by_cc)

    for cc in self.compute_capabilities:
      if cc >= operation.tile_description.minimum_compute_capability and \
         cc <= operation.tile_description.maximum_compute_capability and \
         (cc not in library.SharedMemPerCC or library.SharedMemPerCC[cc] >= library.CalculateSmemUsage(operation)):

        enabled = True
        break

    if not enabled:
      return False

    if len(self.operations_enabled) and not operation.operation_kind in self.operations_enabled:
      return False

    # eliminate duplicates
    if operation.procedural_name() in self.operations_by_name.keys():
      return False

    # Filter based on list of valid substrings
    if len(self.kernel_names):
      name = operation.procedural_name()
      enabled = False

      # compare against the include list
      for name_substr in self.kernel_names:
        if self._filter_string_matches(name_substr, name):
          _LOGGER.debug("Kernel {kernel} included due to filter string '{filt}'.".format(
            kernel = operation.procedural_name(),
            filt = name_substr))
          enabled = True
          break

      # compare against the exclude list
      for name_substr in self.ignore_kernel_names:
        if self._filter_string_matches(name_substr, name):
          _LOGGER.debug("Kernel {kernel} ignored due to filter string '{filt}'.".format(
            kernel = operation.procedural_name(),
            filt = name_substr))
          enabled = False
          break

    if len(self.kernel_filter_list) > 0:
        if self.filter_out_kernels(operation.procedural_name(), self.kernel_filter_list):
          _LOGGER.debug("Kernel {kernel} matched via kernel filter file.".format(kernel = operation.procedural_name()))
          enabled = True
        else:
          _LOGGER.debug("Kernel {kernel} culled due to no match in kernel filter file.".format(kernel = operation.procedural_name()))
          enabled = False


    # TODO: filter based on compute data type
    return enabled


  def append(self, operation):
    '''
      Inserts the operation.

      operation_kind -> configuration_name -> []
    '''

    if self.filter(operation):

      self.selected_kernels.append(operation.procedural_name())

      self.operations_by_name[operation.procedural_name()] = operation

      # add the configuration
      configuration_name = operation.configuration_name()

      # split operations by minimum CC
      min_cc = operation.arch

      if operation.operation_kind not in self.operations.keys():
        self.operations[operation.operation_kind] = {}

      if min_cc not in self.operations[operation.operation_kind]:
        self.operations[operation.operation_kind][min_cc] = {}

      if configuration_name not in self.operations[operation.operation_kind][min_cc].keys():
        self.operations[operation.operation_kind][min_cc][configuration_name] = []

      self.operations[operation.operation_kind][min_cc][configuration_name].append(operation)
      self.operation_count += 1
    else:
      _LOGGER.debug("Culled {} from manifest".format(operation.procedural_name()))


  def emit(self):

    generated_path = os.path.join(self.curr_build_dir, 'generated')

    # create generated/
    if os.path.exists(generated_path):
      shutil.rmtree(generated_path)

    os.mkdir(generated_path)
    for operation_kind, ops in self.operations.items():
      for min_cc, configurations in sorted(ops.items()):
        with EmitOperationKindLibrary(generated_path, min_cc, operation_kind, self.args) as operation_kind_emitter:
          for configuration_name, operations in configurations.items():
            _LOGGER.info("Emitting {config} with {num_ops} operations.".format(
                config = configuration_name, num_ops = len(operations)))
            operation_kind_emitter.emit(configuration_name, operations)

###################################################################################################
