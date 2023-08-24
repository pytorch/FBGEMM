import argparse
import os
import torch
import ai_codesign.nonprod.zhengwangmeta.torchrec_tbe.save_binary_extension as save_binary # @manual

parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
parser.add_argument(
    "--nDev",
    type=int,
    default=1,
    help="number of GPUs",
)
parser.add_argument(
    "--exp",
    type=str,
    default="tbe",
    choices=["tbe", "alltoall", "bwd"],
    help="tbe or alltoall or bwd(fwd+alltoall+bwd)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    nDev = args.nDev

    result_dir = os.path.join(os.environ['HOME'], "tmp/test_1/result")

    if args.exp == "tbe":
        nBatch = 4
        for d_idx in range(nDev):
            for i in range(nBatch):
                torchrec_result_name = os.path.join(result_dir, "torchrec_result_{}_{}.pt".format(d_idx, i))
                kernel_result_name = os.path.join(result_dir, "kernel_result_{}_{}.bin".format(d_idx, i))
                torchrec_result = torch.load(torchrec_result_name)
                kernel_result = save_binary.load_float_tensor(kernel_result_name, torchrec_result.numel())

                are_equal = torch.allclose(torchrec_result.flatten(), kernel_result.flatten())
                assert are_equal, "TBE Test: The result of rank:{} batch:{} are not equal".format(d_idx, i)
        print("All TBE Test Pass")

    elif args.exp == "alltoall":
        nBatch = 4
        for d_idx in range(nDev):
            for i in range(nBatch):
                torchrec_result_name = os.path.join(result_dir, "torchrec_all_to_all_result_{}_{}.pt".format(d_idx, i))
                kernel_result_name = os.path.join(result_dir, "kernel_all_to_all_result_{}_{}.bin".format(d_idx, i))
                torchrec_result = torch.load(torchrec_result_name)
                kernel_result = save_binary.load_float_tensor(kernel_result_name, torchrec_result.numel())

                # print(torchrec_result.view(64,-1), torchrec_result.view(64,-1).shape)
                # for line in torchrec_result.view(64,-1):
                #     print(line)
                # print(kernel_result[0].view(64,-1), kernel_result[0].view(64,-1).shape)
                # for line in kernel_result[0].view(64,-1):
                #     print(line)

                are_equal = torch.allclose(torchrec_result.flatten(), kernel_result.flatten(), atol=1e-5)
                assert are_equal, "TBE + All-to-All Test: The result of rank:{} batch:{} are not equal".format(d_idx, i)
        print("All TBE + All-to-All Test Pass")

    elif args.exp == "bwd":
        for d_idx in range(nDev):
            torchrec_result_name = os.path.join(result_dir, "torchrec_all_to_all_bwd_result_{}.pt".format(d_idx))
            kernel_result_name = os.path.join(result_dir, "kernel_all_to_all_bwd_result_{}.bin".format(d_idx))
            torchrec_result = torch.load(torchrec_result_name)
            kernel_result = save_binary.load_float_tensor(kernel_result_name, torchrec_result.numel())

            are_equal = torch.allclose(torchrec_result.flatten(), kernel_result.flatten(), atol=1e-5)

            # print(torchrec_result[0:400], torchrec_result.sum(), torchrec_result.shape)
            # print(kernel_result[0][0:50], kernel_result.sum(), kernel_result.shape)
            # result_1 = []
            # result_2 = []
            # for i in torchrec_result:
            #     if i.item() != 4.0:
            #         result_1.append(i.item())
            # for i in kernel_result[0]:
            #     if i.item() != 4.0:
            #         result_2.append(i.item())

            # print(result_1, len(result_1), torchrec_result.shape)
            # print(result_2, len(result_2), kernel_result.shape)

            assert are_equal, "TBE + All-to-All + Backward Test: The result of rank:{} are not equal".format(d_idx)
        print("All TBE + All-to-All + Backward Test Pass")
