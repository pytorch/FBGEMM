/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

using namespace std;

void addi(ofstream& of, string i, bool disable = false) {
  if (disable == false)
    of << "\"" + i + "\\t\\n\"" + "\n";
}

struct ISA {
  unsigned avx; // 1, 2 or 3
  string name;
  vector<vector<unsigned>> shapes;
};

int main() {
  bool iaca = false;
  bool disable = false;

  bool fixedA = true, fixedB = true, fixedC = true;

  int eax, ebx, ecx, edx;
  __cpuid(1 /* ecx = vendor string */, eax, ebx, ecx, edx);
  printf("FC16 is %s supported\n", ((ecx & bit_F16C) ? " " : "not"));

  string comma = ",";

  vector<ISA> isa = {
    // {1, "AVX", {{4, 1, 0}, {4, 2, 0}, {4, 3, 0}, {3, 1, 0}, {3, 2, 0}, {3,
    // 3, 0}}},
    { 2, "AVX2",
      { { 1, 1, 0 },
        { 2, 1, 0 },
        { 3, 1, 0 },
        { 4, 1, 0 },
        { 5, 1, 0 },
        { 6, 1, 0 },
        { 7, 1, 0 },
        { 8, 1, 0 },
        { 9, 1, 0 },
        { 10, 1, 0 },
        { 11, 1, 0 },
        { 12, 1, 0 },
        { 13, 1, 0 },
        { 14, 1, 0 },
       }
      }
  };

  // open all files
  ofstream srcfile;
  srcfile.open("FbgemmFP16UKernels.cc");
  srcfile << "#include \"FbgemmFP16UKernels.h\"\n";
  if (iaca)
    srcfile << "#include \"iacaMarks.h\"\n";

  ofstream hdrfile;
  hdrfile.open("FbgemmFP16UKernels.h");

  hdrfile << "#ifndef FBGEMM_UKERNELS\n";
  hdrfile << "#define FBGEMM_UKERNELS\n";
  hdrfile << "#include <cstdint>\n";
  hdrfile << "#include <tuple>\n";
  hdrfile << "#include <vector>\n";
  hdrfile << "#include \"fbgemm/Types.h\"\n";
  hdrfile << "using fp16 = fbgemm2::float16;\n";
  hdrfile << "using fp32 = float;\n";
  hdrfile << "struct GemmParams {uint64_t k; float *A; const fp16 *B;\n"
             "float *beta; uint64_t accum; float *C;  uint64_t ldc;\n"
             "uint64_t b_block_cols; uint64_t b_block_size;};\n";

  std::map<string, string> fptr_typedef;
  fptr_typedef["fp16"] = "";
  fptr_typedef["fp32"] = "";

  unsigned labelId = 0;
#if 1
  for (auto fixedA : {false})
    for (auto fixedB : {false})
      for (auto fixedC : {false})
#else
  for (auto fixedA : {true})
    for (auto fixedB : {true})
      for (auto fixedC : {true})
#endif
        for (auto s : isa) {
          vector<vector<unsigned>>& ukernel_shape = s.shapes;

          vector<string> funcname(ukernel_shape.size()),
              fheader(ukernel_shape.size());
          string fargs;

          for (auto fp16 : {true}) {
            string B_type = ((fp16) ? "fp16" : "fp32");
            string prefix = s.name + /*"_" + B_type */ + "_" + "fA" +
                to_string(fixedA) + "fB" + to_string(fixedB) + "fC" +
                to_string(fixedC);
            cout << "Generating code for " << s.name << " " << B_type << "\n";

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              printf(
                  "shape: %d x %d * 32\n",
                  ukernel_shape[k][0],
                  ukernel_shape[k][1]);

              string p1 = "GemmParams *gp";

              funcname[k] = "gemmkernel_" + to_string(ukernel_shape[k][0]) +
                  "x" + to_string(ukernel_shape[k][1]) + "_";
              funcname[k] += prefix;

              fargs = "(" + p1 + ")";

              fheader[k] =
                  "void __attribute__ ((noinline)) " + funcname[k] + fargs;
              srcfile << fheader[k] << "\n";
              srcfile << "{\n";

              unsigned last_free_ymmreg = 0;
              // produce register block of C
              vector<vector<string>> vCtile(ukernel_shape[k][0]);
              for (auto r = 0; r < ukernel_shape[k][0]; r++)
                for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                  vCtile[r].push_back("ymm" + to_string(last_free_ymmreg));
                  last_free_ymmreg++;
                }
              assert(last_free_ymmreg <= 14);

              string vAtmp = "ymm" + to_string(last_free_ymmreg++);
              // produce register block of B col
              assert(ukernel_shape[k][1] == 1);
              vector<string> vBcol(ukernel_shape[k][1]);

              for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                vBcol[c] = ("ymm" + to_string(last_free_ymmreg));
                last_free_ymmreg++;
              }

              assert(last_free_ymmreg <= 16);

              srcfile << "asm volatile\n";
              srcfile << "(\n";

              srcfile << "#if !defined(__clang__)" << "\n";
              addi(srcfile, "mov r14, %[gp]");
              srcfile << "#else\n";
              addi(srcfile, "mov %[gp], %%r14");
              addi(srcfile, ".intel_syntax noprefix");
              srcfile << "#endif\n";

              srcfile << "\n// Copy parameters\n";
              srcfile << "// k\n";
              addi(srcfile, "mov r8, [r14 + 0]");
              srcfile << "// A\n";
              addi(srcfile, "mov r9, [r14 + 8]");
              srcfile << "// B\n";
              addi(srcfile, "mov r10, [r14 + 16]");
              srcfile << "// beta\n";
              addi(srcfile, "mov r15, [r14 + 24]");
              srcfile << "// accum\n";
              addi(srcfile, "mov rdx, [r14 + 32]");
              srcfile << "// C\n";
              addi(srcfile, "mov r12, [r14 + 40]");
              srcfile << "// ldc\n";
              addi(srcfile, "mov r13, [r14 + 48]");
              srcfile << "// b_block_cols\n";
              addi(srcfile, "mov rdi, [r14 + 56]");
              srcfile << "// b_block_size\n";
              addi(srcfile, "mov rsi, [r14 + 64]");
              srcfile << "// Make copies of A and C\n";
              addi(srcfile, "mov rax, r9");
              addi(srcfile, "mov rcx, r12");
              srcfile << "\n\n";

              addi(srcfile, "mov rbx, 0");

              string exitlabel = "L_exit%=";
              string label2 = "loop_outter%=";
              addi(srcfile, label2 + ":");
              addi(srcfile, "mov r14, 0");

              // set all vCtile regs to zeros
              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  addi(
                      srcfile,
                      "vxorps " + vCtile[r][c] + "," + vCtile[r][c] + "," +
                          vCtile[r][c]);
                }
              }

              // start marker
              if (iaca) {
                addi(srcfile, "mov ebx, 111");
                addi(srcfile, ".byte 0x64, 0x67, 0x90");
              }

              srcfile << "\n";

              if (ukernel_shape[k][0] <= 13) {
                addi(srcfile, "vcvtph2ps ymm15, XMMWORD PTR [r10 + 0]");
                addi(srcfile, "mov r11, 16");
              } else {
                addi(srcfile, "mov r11, 0");
              }

              srcfile << "\n";
              string label = "loop_inner%=";
              addi(srcfile, label + ":");
              srcfile << "\n";

              if (ukernel_shape[k][0] <= 13) {
                auto a_offset = 0, unroll_factor = 2;
                for (auto u = 0; u < unroll_factor; u++) {
                  string breg = (u == 0) ? "ymm14" : "ymm15";
                  string breg_rev = (u == 0) ? "ymm15" : "ymm14";

                  addi(srcfile, "vcvtph2ps " + breg +
                                    ",XMMWORD PTR [r10 + r11 + " +
                                    to_string(u * 16) + "]");
                  addi(srcfile, "inc r14");
                  for (auto r = 0; r < vCtile.size(); r++) {
                    addi(srcfile, "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" +
                                      to_string(a_offset) + "]");
                    addi(srcfile, "vfmadd231ps " + vCtile[r][0] + "," +
                                      breg_rev + "," + vAtmp);
                    if (u == 1 && r == vCtile.size() / 2)
                      addi(srcfile, "add r11, 32");
                    a_offset += 4;
                  }
                  if (u < unroll_factor - 1) {
                    addi(srcfile, "cmp r14, r8");
                    addi(srcfile, "jge " + exitlabel);
                  }
                }

                addi(srcfile, "add r9," + to_string(a_offset));
                addi(srcfile, "cmp r14, r8");
                addi(srcfile, "jl " + label);

                srcfile << "\n";

                addi(srcfile, exitlabel + ":");
              } else {
                addi(srcfile,
                     "vcvtph2ps " + vBcol[0] + ",XMMWORD PTR [r10 + r11]");
                for (auto r = 0; r < vCtile.size(); r++) {
                  addi(srcfile, "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" +
                                    to_string(4 * r) + "]");
                  addi(srcfile, "vfmadd231ps " + vCtile[r][0] + "," + vBcol[0] +
                                    "," + vAtmp);
                }

                addi(srcfile, "add r9," + to_string(4 * ukernel_shape[k][0]),
                     fixedA); // move A ptr
                addi(srcfile, "add r11, 16");

                addi(srcfile, "inc r14");
                addi(srcfile, "cmp r14, r8");
                addi(srcfile, "jl " + label);
              }

              addi(srcfile, "add r10, rsi");
              srcfile << "\n";

              // end marker
              if (iaca) {
                addi(srcfile, "mov ebx, 222");
                addi(srcfile, ".byte 0x64, 0x67, 0x90");
              }


              addi(srcfile, "cmp rdx, 1");
              addi(srcfile, "je L_accum%=");
              srcfile << "// Dump C\n";

              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                    to_string(32 * c) + "], " + vCtile[r][c],
                       fixedC);
                }
                addi(srcfile, "add r12, r13", fixedC); // move C ptr
              }
              addi(srcfile, "jmp L_done%=");

              srcfile << "\n\n";
              addi(srcfile, "L_accum%=:");
              srcfile << "// Dump C with accumulate\n";

              string r_spare = (s.avx == 1) ? "ymm14" : "ymm15";
              addi(srcfile,
                   "vbroadcastss " + r_spare + string(",DWORD PTR [r15]"),
                   fixedC);
              // store out C
              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  switch (s.avx) {
                  case 1:
                    addi(srcfile,
                         string("vmulps ymm15, ") + r_spare + comma +
                             "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                         fixedC);
                    addi(srcfile, "vaddps " + vCtile[r][c] + "," +
                                      vCtile[r][c] + "," + "ymm15",
                         fixedC);
                    break;
                  case 2:
                    addi(srcfile,
                         "vfmadd231ps " + vCtile[r][c] + "," + r_spare + "," +
                             "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                         fixedC);
                    break;
                  default:
                    assert(0);
                  }
                  addi(srcfile, "vmovups YMMWORD PTR [r12 + " +
                                    to_string(32 * c) + "], " + vCtile[r][c],
                       fixedC);
                }
                addi(srcfile, "add r12, r13", fixedC); // move C ptr
              }

              srcfile << "\n";
              addi(srcfile, "L_done%=:");

              srcfile << "\n// next outer iteration\n";
              // C
              addi(srcfile, "add rcx, " + to_string(32 * ukernel_shape[k][1]),
                   fixedC);
              addi(srcfile, "mov r12, rcx", fixedC);
              // A
              addi(srcfile, "mov r9, rax");

              addi(srcfile, "inc rbx");
              addi(srcfile, "cmp rbx, rdi");
              addi(srcfile, "jl " + label2);

              // output
              srcfile << ":\n";
              // input
              srcfile << ":\n";
              srcfile << "[gp] \"rm\" (gp)\n";

              // clobbered
              srcfile
                  << (string) ": \"r8\", \"r9\", \"r10\", \"r11\", \"r15\", " +
                         (string) " \"r13\", \"r14\",\n" +
                         (string) "\"rax\", \"rcx\", "
                                  "\"rdx\", \"rsi\", \"rdi\", \"rbx\", "
                                  "\"r12\", \"memory\"" +
                         (string) "\n";
              srcfile << ");\n";
              srcfile << "}\n";
            }

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              hdrfile << fheader[k] << ";\n";
            }

            fptr_typedef[B_type] =
                "typedef void (* funcptr_" + B_type + ") " + fargs;
          }
        }

  srcfile.close();
  hdrfile << fptr_typedef["fp16"] << ";\n";
  hdrfile << fptr_typedef["fp32"] << ";\n";
  hdrfile << "#endif\n";
  hdrfile.close();
}
