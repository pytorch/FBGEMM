# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FLOPs and memory traffic analysis for IKBO LCE kernel variants.
#
# Optimization progression:
#   Baseline:        W @ E, single batched matmul with pre-broadcast user embeddings
#   Decomposition:   W_cand @ E_cand + W_user @ E_user[u(b)], avoids redundant user compute
#   K-dim alignment: Same as decomposition but K padded to tile boundary
#   Kernel fusion:   Triton kernel fuses cand matmul + user broadcast add (user matmul still cuBLAS)
#   TLX:             Single persistent kernel fuses both matmuls + broadcast add via warp specialization

import torch

DEVICE = "cuda"
DTYPE = torch.float16

# Representative realistic dimensions.
B, M, N, K_USER, K_CAND = 1024, 433, 256, 1178, 866
CAND_TO_USER_RATIO = 70


def _prepare_default_inputs():
    """Create default test inputs with fixed candidate-to-user ratio."""
    num_users = (B + CAND_TO_USER_RATIO - 1) // CAND_TO_USER_RATIO
    cand_to_user_index = (torch.arange(B, device=DEVICE) // CAND_TO_USER_RATIO).int()

    cw_cand = torch.randn((M, K_CAND), device=DEVICE, dtype=DTYPE)
    cw_user = torch.randn((M, K_USER), device=DEVICE, dtype=DTYPE)
    e_cand = torch.randn((B, K_CAND, N), device=DEVICE, dtype=DTYPE)
    e_user = torch.randn((num_users, K_USER, N), device=DEVICE, dtype=DTYPE)

    cw = torch.cat((cw_user, cw_cand), dim=1)
    e = torch.cat((torch.index_select(e_user, 0, cand_to_user_index), e_cand), dim=1)

    return cw_cand, cw_user, e_cand, e_user, cand_to_user_index, cw, e


# ============================================================
# FLOPs utilities (unit: GFLOPS, rounded to 0.01)
# ============================================================
# All variants perform the same arithmetic: two matmuls + one element-wise add.
# FLOPs are identical across variants; only memory access patterns differ.


def get_lce_flops(compression_w, embeddings):
    """Baseline: W[M,K] @ E[B,K,N] -> 2*B*M*K*N FLOPs."""
    M, K = compression_w.shape
    B, K2, N = embeddings.shape
    assert K == K2
    return round(2 * B * M * K * N / 1e9, 2)


def get_decomposed_lce_flops(
    compression_w_cand, compression_w_user, embeddings_cand, embeddings_user
):
    """Decomposed: cand matmul + user matmul + element-wise add.

    Returns (cand_gflops, user_gflops, add_gflops, total_gflops).
    """
    M, K_CAND = compression_w_cand.shape
    B, K_CAND2, N = embeddings_cand.shape
    assert K_CAND == K_CAND2
    K_USER = compression_w_user.shape[1]
    U = embeddings_user.shape[0]

    cand_matmul_gflops = round(2 * B * M * K_CAND * N / 1e9, 2)
    user_matmul_gflops = round(2 * U * M * K_USER * N / 1e9, 2)
    add_gflops = round(B * M * N / 1e9, 2)
    total_gflops = round(cand_matmul_gflops + user_matmul_gflops + add_gflops, 2)
    return cand_matmul_gflops, user_matmul_gflops, add_gflops, total_gflops


# ============================================================
# Bytes utilities (unit: MB, integer)
# ============================================================
# Memory traffic differs across variants due to intermediate buffers
# and kernel fusion boundaries.


def get_lce_bytes(compression_w, embeddings):
    """Baseline: single batched matmul.

    Read:  W[M,K] + E[B,K,N]
    Write: out[B,M,N]
    """
    M, K = compression_w.shape
    B, K2, N = embeddings.shape
    assert K == K2
    read = (M * K + B * K2 * N) * 2 // 1024**2
    write = B * M * N * 2 // 1024**2
    return read, write, read + write


def get_decomposed_lce_bytes(
    compression_w_cand,
    compression_w_user,
    embeddings_cand,
    embeddings_user,
    cand_to_user_index,
):
    """Decomposed: four separate PyTorch ops, each a distinct kernel launch.

    Op 1 (cand matmul):  read W_cand + E_cand,            write cand_res[B,M,N]
    Op 2 (user matmul):  read W_user + E_user,             write user_res[U,M,N]
    Op 3 (index_select): read user_res[U,M,N] + idx[B],    write broadcast[B,M,N]
    Op 4 (add):          read cand_res[B,M,N] + broadcast,  write out[B,M,N]
    """
    M = compression_w_cand.shape[0]
    B = embeddings_cand.shape[0]
    N = embeddings_cand.shape[2]
    U = embeddings_user.shape[0]

    cand_read, cand_write, _ = get_lce_bytes(compression_w_cand, embeddings_cand)
    user_read, user_write, _ = get_lce_bytes(compression_w_user, embeddings_user)

    index_select_read = (U * M * N * 2 + B * 4) // 1024**2
    index_select_write = (B * M * N * 2) // 1024**2

    add_read = (2 * B * M * N * 2) // 1024**2
    add_write = (B * M * N * 2) // 1024**2

    total_read = cand_read + user_read + index_select_read + add_read
    total_write = cand_write + user_write + index_select_write + add_write
    return total_read, total_write, total_read + total_write


def get_kernel_fusion_bytes(
    compression_w_cand,
    compression_w_user,
    embeddings_cand,
    embeddings_user,
    cand_to_user_index,
):
    """Kernel fusion (triton_ikbo_lce): user matmul via cuBLAS, cand+add fused in Triton.

    cuBLAS:  read W_user + E_user,                  write user_res[U,M,N]
    Triton:  read W_cand + E_cand + user_res + idx,  write out[B,M,N]

    Saves index_select + add intermediate traffic vs decomposed.
    Still materializes user_res[U,M,N] between cuBLAS and Triton.
    """
    M, K_CAND = compression_w_cand.shape
    K_USER = compression_w_user.shape[1]
    B, _, N = embeddings_cand.shape
    U = embeddings_user.shape[0]

    read = (
        M * K_CAND + M * K_USER + B * K_CAND * N + U * K_USER * N + U * M * N
    ) * 2 + B * 4
    write = (U * M * N + B * M * N) * 2
    read_mb = read // 1024**2
    write_mb = write // 1024**2
    return read_mb, write_mb, read_mb + write_mb


def _fmt_bytes(label, read, write, total):
    return f"  {label:<20s}  read={read:>5d} MB  write={write:>5d} MB  total={total:>5d} MB"


def _fmt_flops(label, breakdown):
    if isinstance(breakdown, tuple):
        cand, user, add, total = breakdown
        return f"  {label:<20s}  cand={cand:>7.2f}  user={user:>7.2f}  add={add:>7.2f}  total={total:>7.2f} GFLOPS"
    return f"  {label:<20s}  total={breakdown:>7.2f} GFLOPS"


def print_flops_bytes():
    torch.manual_seed(0)
    (
        compression_w_cand,
        compression_w_user,
        embeddings_cand,
        embeddings_user,
        cand_to_user_index,
        compression_w,
        embeddings,
    ) = _prepare_default_inputs()

    M = compression_w_cand.shape[0]
    B = embeddings_cand.shape[0]
    N = embeddings_cand.shape[2]
    U = embeddings_user.shape[0]
    K_CAND = compression_w_cand.shape[1]
    K_USER = compression_w_user.shape[1]
    K_TOTAL = compression_w.shape[1]

    print(
        f"Dimensions: B={B}, M={M}, N={N}, K_cand={K_CAND}, K_user={K_USER}, K_total={K_TOTAL}, U={U}"
    )
    print()
    print("Memory traffic:")
    print(_fmt_bytes("Baseline", *get_lce_bytes(compression_w, embeddings)))
    print(
        _fmt_bytes(
            "Decomposed",
            *get_decomposed_lce_bytes(
                compression_w_cand,
                compression_w_user,
                embeddings_cand,
                embeddings_user,
                cand_to_user_index,
            ),
        )
    )
    print(
        _fmt_bytes(
            "Kernel fusion",
            *get_kernel_fusion_bytes(
                compression_w_cand,
                compression_w_user,
                embeddings_cand,
                embeddings_user,
                cand_to_user_index,
            ),
        )
    )
    print()
    print("FLOPs:")
    print(_fmt_flops("Baseline", get_lce_flops(compression_w, embeddings)))
    print(
        _fmt_flops(
            "Decomposed",
            get_decomposed_lce_flops(
                compression_w_cand, compression_w_user, embeddings_cand, embeddings_user
            ),
        )
    )


def main():
    print_flops_bytes()


if __name__ == "__main__":
    main()
