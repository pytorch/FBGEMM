# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors[56]

import torch
import triton  # noqa: F401
from fbgemm_gpu.experimental.fb.gen_ai.moe import gather_scale_dense_tokens
from triton.testing import do_bench


def bench_gather_scale_dense_tokens(E: int, T: int, D: int):
    x = torch.randn((T, D), dtype=torch.bfloat16, device="cuda").abs()
    expert_indices = torch.randint(0, E, (T,), device="cuda")
    token_indices = torch.randperm(T, device="cuda")
    scores = torch.rand((E, T), dtype=torch.bfloat16, device="cuda")

    def torch_fn():
        shuffled_x = torch.index_select(x, dim=0, index=token_indices)
        shuffled_scores = torch.index_select(scores, dim=1, index=token_indices)
        shuffled_selected_scores = torch.gather(
            shuffled_scores, dim=0, index=expert_indices.view(1, T)
        )
        ref_output = shuffled_x * shuffled_selected_scores.view(-1, 1)
        return ref_output

    torch_output = torch_fn()

    scores_TE = scores.transpose(0, 1).contiguous()

    def triton_fn():
        test_output = gather_scale_dense_tokens(
            x, token_indices, expert_indices, scores_TE
        )
        return test_output

    test_output = triton_fn()

    torch.testing.assert_close(torch_output, test_output)

    # Run benchmark
    data_size_in_gigabytes = T * D * 2 * 2 / 1e9
    torch_time = do_bench(torch_fn, rep=1000) * 1e3
    torch_bw = data_size_in_gigabytes / (torch_time / 1e6)
    triton_time = do_bench(triton_fn, rep=1000) * 1e3
    triton_bw = data_size_in_gigabytes / (triton_time / 1e6)
    print(
        f"Running benchmark, {E=:3d}, {T=:4d}, {D=:4d}: "
        f"Torch time: {torch_time:10.3f} us. Bandwidth: {torch_bw:10.3f} GB/s. "
        f"Triton time: {triton_time:10.3f} us. Bandwidth: {triton_bw:10.3f} GB/s."
    )


def main():
    for E in [2, 4, 8, 16]:
        for T in [1, 128, 2048, 4096]:
            for D in [5120]:
                bench_gather_scale_dense_tokens(E, T, D)


if __name__ == "__main__":
    main()
