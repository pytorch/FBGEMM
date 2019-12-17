/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Arguments to code gen
 * --prefetch[=n] - use prefetch in code generation, n == prefetch len
 * --fp32         - generate code for FBGEMM32 (SGEMM)
 */

#include <assert.h>
#include <cpuid.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>

using namespace std;

void addi(ofstream& of, string i, bool disable = false) {
  if (disable == false)
    of << "      \"" + i + "\\t\\n\"" + "\n";
}

struct ISA {
  enum class isaType { avx, avx2, avx512, avx512_256 };
  isaType iset;
  string name;
  vector<vector<unsigned>> shapes;
};

enum class DataType { Float32, Float16, BFloat16 };

std::array<std::pair<DataType, std::string>, 2> types_to_gen = {
    std::make_pair(DataType::Float32, "FP32"),
    std::make_pair(DataType::Float16, "FP16")};

constexpr int cache_line_size = 64;
constexpr int prefetch_len_default = cache_line_size * 12;

int parseArgumentInt(
    int argc,
    const char* argv[],
    const char* arg,
    int non_exist_val,
    int def_val) {
  int val = non_exist_val;
  int arg_len = strlen(arg);
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      val = (*(ptr + arg_len) == '=') ? atoi(ptr + arg_len + 1) : def_val;
      break;
    }
  }
  return val;
}

bool parseArgumentBool(
    int argc,
    const char* argv[],
    const char* arg,
    bool def_val) {
  int arg_len = strlen(arg);
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      return true;
    }
  }
  return def_val;
}

int main(int argc, const char* argv[]) {
  bool iaca = false;
  bool disable = false;
  unordered_set<string> enabledDataType;

  // Always generate FP16
  enabledDataType.insert("FP16");
  if (parseArgumentBool(argc, argv, "--fp32", false)) {
    enabledDataType.insert("FP32");
  }
  const int prefetch_len =
      parseArgumentInt(argc, argv, "--prefetch", 0, prefetch_len_default);

  bool fixedA = true, fixedB = true, fixedC = true;

  int eax, ebx, ecx, edx;
  __cpuid(1 /* ecx = vendor string */, eax, ebx, ecx, edx);
  printf("FC16 is %s supported\n", ((ecx & bit_F16C) ? " " : "not"));

  static const string license =
      "/*\n"
      " * Copyright (c) Facebook, Inc. and its affiliates.\n"
      " * All rights reserved.\n"
      " * This source code is licensed under the BSD-style license found in the\n"
      " * LICENSE file in the root directory of this source tree.\n"
      " */\n";

  string comma = ",";

  vector<ISA> isa = {
      // {1, "AVX", {{4, 1, 0}, {4, 2, 0}, {4, 3, 0}, {3, 1, 0}, {3, 2, 0}, {3,
      // 3, 0}}},
      {ISA::isaType::avx2,
       "Avx2",
       {
           // 4x3 register layout
           // {1, 3, 0},
           // {2, 3, 0},
           // {3, 3, 0},
           // {4, 3, 0},

           // 6x2 register layout
           {1, 2, 0},
           {2, 2, 0},
           {3, 2, 0},
           {4, 2, 0},
           {5, 2, 0},
           {6, 2, 0},

           // 14x1 register layout
           // {1, 1, 0},
           // {2, 1, 0},
           // {3, 1, 0},
           // {4, 1, 0},
           // {5, 1, 0},
           // {6, 1, 0},
           // {7, 1, 0},
           // {8, 1, 0},
           // {9, 1, 0},
           // {10, 1, 0},
           // {11, 1, 0},
           // {12, 1, 0},
           // {13, 1, 0},
           // {14, 1, 0},
       }},
      {ISA::isaType::avx512,
       "Avx512",
       {
           // 14x2 register layout
           {1, 2, 0},
           {2, 2, 0},
           {3, 2, 0},
           {4, 2, 0},
           {5, 2, 0},
           {6, 2, 0},
           {7, 2, 0},
           {8, 2, 0},
           {9, 2, 0},
           {10, 2, 0},
           {11, 2, 0},
           {12, 2, 0},
           {13, 2, 0},
           {14, 2, 0},
       }},
      {ISA::isaType::avx512_256,
       "Avx512_256",
       {
           // 14x2 register layout
           // Implemented by AVX2
           //{1, 2, 0},
           //{2, 2, 0},
           //{3, 2, 0},
           //{4, 2, 0},
           //{5, 2, 0},
           //{6, 2, 0},
           {7, 2, 0},
           {8, 2, 0},
           {9, 2, 0},
           {10, 2, 0},
           {11, 2, 0},
           {12, 2, 0},
           {13, 2, 0},
           {14, 2, 0},
       }}};

  for (auto& d_type : types_to_gen) {
    if (enabledDataType.count(d_type.second) == 0) {
      continue;
    }
    for (auto s : isa) {
      bool const isFp16 = d_type.first != DataType::Float32;
      string const B_type = [&]() {
        if (d_type.first == DataType::Float32)
          return "fp32";
        if (d_type.first == DataType::Float16)
          return "fp16";
      }();

      string isa_file_name = "Fbgemm" + d_type.second + "UKernels" + s.name;

      // open all files
      ofstream srcfile;
      srcfile.open(isa_file_name + ".cc");
      srcfile << license;
      srcfile << "#include \"./" + isa_file_name + ".h\"\n\n";
      srcfile << "namespace fbgemm {\n\n";
      if (iaca) {
        srcfile << "#include \"iacaMarks.h\"\n";
      }

      ofstream hdrfile;
      hdrfile.open(isa_file_name + ".h");
      hdrfile << license;

      hdrfile << "#pragma once\n";
      hdrfile << "#include <cstdint>\n";
      hdrfile << "#include \"fbgemm/Types.h\"\n\n";
      hdrfile << "#include \"fbgemm/FbgemmBuild.h\"\n\n";
      hdrfile << "#include \"./Fbgemm" + d_type.second + "Common.h\"\n\n";
      hdrfile << "namespace fbgemm {\n\n";

      unsigned labelId = 0;

      bool fixedA = false, fixedB = false, fixedC = false;

      vector<vector<unsigned>>& ukernel_shape = s.shapes;

      vector<string> funcname(ukernel_shape.size()),
          fheader(ukernel_shape.size());
      string fargs;

      string prefix = s.name + "_" + B_type + "_" + "fA" + to_string(fixedA) +
          "fB" + to_string(fixedB) + "fC" + to_string(fixedC);
      cout << "Generating code for " << s.name << " " << B_type << "\n";

      string vec_reg_prefix = s.iset == ISA::isaType::avx512 ? "zmm" : "ymm";
      int num_vec_regs = s.iset == ISA::isaType::avx2 ? 16 : 32;
      int vec_len_in_bytes = s.iset == ISA::isaType::avx512 ? 64 : 32;

      for (unsigned k = 0; k < ukernel_shape.size(); k++) {
        printf(
            "shape: %d x %d * 32\n", ukernel_shape[k][0], ukernel_shape[k][1]);

        string p1 = "GemmParams" + d_type.second + "* gp";

        funcname[k] = "gemmkernel_" + to_string(ukernel_shape[k][0]) + "x" +
            to_string(ukernel_shape[k][1]) + "_";
        funcname[k] += prefix;

        fargs = "(" + p1 + ")";

        fheader[k] = "void NOINLINE\n" + funcname[k] + fargs;
        srcfile << fheader[k] << " {\n";

        unsigned last_free_vecreg = 0;
        // produce register block of C
        vector<vector<string>> vCtile(ukernel_shape[k][0]);
        for (auto r = 0; r < ukernel_shape[k][0]; r++)
          for (auto c = 0; c < ukernel_shape[k][1]; c++) {
            vCtile[r].push_back(vec_reg_prefix + to_string(last_free_vecreg));
            last_free_vecreg++;
          }
        assert(last_free_vecreg <= num_vec_regs - 2);

        string vAtmp = vec_reg_prefix + to_string(last_free_vecreg++);
        // produce register block of B col
        vector<string> vBcol(ukernel_shape[k][1]);

        for (auto c = 0; c < ukernel_shape[k][1]; c++) {
          vBcol[c] = (vec_reg_prefix + to_string(last_free_vecreg));
          last_free_vecreg++;
        }
        assert(last_free_vecreg <= num_vec_regs);

        srcfile << "  asm volatile(\n";

        srcfile << "#if !defined(__clang__)"
                << "\n";
        addi(srcfile, "mov r14, %[gp]");
        srcfile << "#else\n";
        addi(srcfile, "mov %[gp], %%r14");
        addi(srcfile, ".intel_syntax noprefix");
        srcfile << "#endif\n";

        srcfile << "\n";
        srcfile << "      // Copy parameters\n";
        srcfile << "      // k\n";
        addi(srcfile, "mov r8, [r14 + 0]");
        srcfile << "      // A\n";
        addi(srcfile, "mov r9, [r14 + 8]");
        srcfile << "      // B\n";
        addi(srcfile, "mov r10, [r14 + 16]");
        srcfile << "      // beta\n";
        string r_spare = vec_reg_prefix +
            to_string(num_vec_regs - (s.iset == ISA::isaType::avx ? 2 : 1));
        addi(
            srcfile,
            "vbroadcastss " + r_spare + string(",DWORD PTR [r14 + 24]"),
            fixedC);
        srcfile << "      // C\n";
        addi(srcfile, "mov r12, [r14 + 32]");
        srcfile << "      // ldc\n";
        addi(srcfile, "mov r13, [r14 + 40]");
        srcfile << "      // b_block_cols\n";
        addi(srcfile, "mov rdi, [r14 + 48]");
        srcfile << "      // b_block_size\n";
        addi(srcfile, "mov rsi, [r14 + 56]");
        srcfile << "\n";
        srcfile << "      // Make copies of A and C\n";
        addi(srcfile, "mov rax, r9");
        addi(srcfile, "mov rcx, r12");
        srcfile << "\n";

        addi(srcfile, "mov rbx, 0");

        string label_outer = "loop_outter%=";
        addi(srcfile, label_outer + ":");
        addi(srcfile, "mov r14, 0");

        string label_inner = "loop_inner%=";
        string label_zero = "zero_regs%=";
        string r_spare_cmp = "xmm" +
            to_string(num_vec_regs - (s.iset == ISA::isaType::avx ? 2 : 1));
        addi(srcfile, "vxorps xmm0, xmm0, xmm0");
        addi(srcfile, "vcomiss " + r_spare_cmp + ", xmm0");
        addi(srcfile, "jz " + label_zero);

        srcfile << "\n";
        srcfile << "      // Setup values with beta multiplication\n";
        string r_last = vec_reg_prefix + to_string(num_vec_regs - 1);
        // store out C
        for (auto r = 0; r < vCtile.size(); r++) {
          if (r > 0) {
            addi(srcfile, "add r12, r13", fixedC); // move C ptr
          }
          for (auto c = 0; c < vCtile[r].size(); c++) {
            switch (s.iset) {
              case ISA::isaType::avx:
              case ISA::isaType::avx2:
              case ISA::isaType::avx512:
              case ISA::isaType::avx512_256:
                if (prefetch_len &&
                    ((vec_len_in_bytes * c) % cache_line_size == 0)) {
                  addi(
                      srcfile,
                      "prefetcht0 [r12 + " +
                          to_string(
                              vec_len_in_bytes * ukernel_shape[k][1] +
                              c * cache_line_size) +
                          "]",
                      fixedC);
                }
                addi(
                    srcfile,
                    "vmulps " + vCtile[r][c] + ", " + r_spare + ", " +
                        "[r12 + " + to_string(vec_len_in_bytes * c) + "]",
                    fixedC);
                break;
              default:
                assert(0);
            }
          }
        }
        if (vCtile.size() > 1) {
          addi(srcfile, "mov r12, rcx");
        }
        addi(srcfile, "jmp " + label_inner);

        srcfile << "\n";
        addi(srcfile, label_zero + ":");
        srcfile << "\n";

        // set all vCtile regs to zeros
        for (auto r = 0; r < vCtile.size(); r++) {
          if (r > 0) {
            addi(srcfile, "add r12, r13", fixedC); // move C ptr
          }
          for (auto c = 0; c < vCtile[r].size(); c++) {
            if (prefetch_len &&
                ((vec_len_in_bytes * c) % cache_line_size == 0)) {
              addi(
                  srcfile,
                  "prefetcht0 [r12 + " + std::to_string(c * prefetch_len) + "]",
                  fixedC);
            }
            addi(
                srcfile,
                "vxorps " + vCtile[r][c] + ", " + vCtile[r][c] + ", " +
                    vCtile[r][c]);
          }
        }
        if (vCtile.size() > 1) {
          addi(srcfile, "mov r12, rcx");
        }

        // start marker
        if (iaca) {
          addi(srcfile, "mov ebx, 111");
          addi(srcfile, ".byte 0x64, 0x67, 0x90");
        }

        srcfile << "\n";
        addi(srcfile, label_inner + ":");
        srcfile << "\n";

        auto const B_load = [&](int c) {
          if (d_type.first == DataType::Float32) {
            addi(
                srcfile,
                "vmovups " + vBcol[c] + "," +
                    (s.iset == ISA::isaType::avx512 ? "ZMM" : "YMM") +
                    "WORD PTR [r10 + " + to_string(vec_len_in_bytes * c) + "]");
          } else if (d_type.first == DataType::Float16) {
            addi(
                srcfile,
                "vcvtph2ps " + vBcol[c] + "," +
                    (s.iset == ISA::isaType::avx512 ? "YMM" : "XMM") +
                    "WORD PTR [r10 + " + to_string(vec_len_in_bytes / 2 * c) +
                    "]");
          }
        };

        for (int c = 0; c < vCtile[0].size(); c++) {
          B_load(c);
          if (prefetch_len && ((vec_len_in_bytes * c) % cache_line_size == 0)) {
            addi(
                srcfile,
                "prefetcht0 [r10 + " +
                    to_string(vec_len_in_bytes * c + prefetch_len) + "]",
                fixedC);
          }
        }

        for (int r = 0; r < vCtile.size(); r++) {
          addi(
              srcfile,
              "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" + to_string(4 * r) +
                  "]");
          for (int c = 0; c < vCtile[0].size(); c++) {
            addi(
                srcfile,
                "vfmadd231ps " + vCtile[r][c] + "," + vBcol[c] + "," + vAtmp);
          }
        }

        addi(
            srcfile,
            "add r9," + to_string(4 * ukernel_shape[k][0]),
            fixedA); // move A ptr

        addi(
            srcfile,
            "add r10," +
                to_string(
                    (vec_len_in_bytes >> (int)isFp16) * ukernel_shape[k][1]),
            fixedA); // move A ptr

        addi(srcfile, "inc r14");
        addi(srcfile, "cmp r14, r8");
        addi(srcfile, "jl " + label_inner);

        srcfile << "\n";

        // end marker
        if (iaca) {
          addi(srcfile, "mov ebx, 222");
          addi(srcfile, ".byte 0x64, 0x67, 0x90");
        }

        srcfile << "      // Dump C\n";

        for (auto r = 0; r < vCtile.size(); r++) {
          if (r > 0) {
            addi(srcfile, "add r12, r13", fixedC); // move C ptr
          }
          for (auto c = 0; c < vCtile[r].size(); c++) {
            addi(
                srcfile,
                "vmovups " + vec_reg_prefix + "word PTR [r12 + " +
                    to_string(vec_len_in_bytes * c) + "], " + vCtile[r][c],
                fixedC);
          }
        }

        srcfile << "\n      // next outer iteration\n";
        // C
        addi(
            srcfile,
            "add rcx, " + to_string(vec_len_in_bytes * ukernel_shape[k][1]),
            fixedC);
        addi(srcfile, "mov r12, rcx", fixedC);
        // A
        addi(srcfile, "mov r9, rax");

        addi(srcfile, "inc rbx");
        addi(srcfile, "cmp rbx, rdi");
        addi(srcfile, "jl " + label_outer);

        // output
        srcfile << "      :\n";
        // input
        srcfile << "      : [gp] \"rm\"(gp)\n";

        // clobbered
        srcfile << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                   "        \"r11\",\n        \"r13\",\n"
                   "        \"r14\",\n        \"rax\",\n        \"rcx\",\n"
                   "        \"rsi\",\n        \"rdi\",\n"
                   "        \"rbx\",\n        \"r12\",\n"
                   "        \"memory\");\n";
        srcfile << "}\n";
      }

      for (unsigned k = 0; k < ukernel_shape.size(); k++) {
        hdrfile << fheader[k] << ";\n";
      }

      srcfile << "\n} // namespace fbgemm\n";
      srcfile.close();
      hdrfile << "\n} // namespace fbgemm\n";
      hdrfile.close();
    } // isa
  }
}
