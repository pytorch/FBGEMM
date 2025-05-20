# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Finds and sets GNU_FH2_IEEE compilation flags
################################################################################

INCLUDE(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("
    #include <arm_neon.h>
    int main() {
        float f = 1.0f;
        uint16_t h = __gnu_f2h_ieee(f);
        return 0;
    }
" HAVE_GNU_F2H_IEEE)
