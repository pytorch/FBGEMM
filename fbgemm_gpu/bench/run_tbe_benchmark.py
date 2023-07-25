#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import subprocess


def run(args):
    with open(args.shapes_file, "r") as f:
        shapes = json.load(f)

    num_embeddings_list = ",".join([str(shape[0]) for shape in shapes])
    embedding_dims_list = ",".join([str(shape[1]) for shape in shapes])

    cmds = [
        args.python,
        args.benchmark_path,
        args.benchmark_cmd,
        "--batch-size",
        str(args.batch_size),
        "--bag-size-list",
        str(args.bag_size),
        "--embedding-dim-list",
        embedding_dims_list,
        "--num-embeddings-list",
        num_embeddings_list,
        "--weights-precision",
        args.weights_precision,
        "--output-dtype",
        args.output_dtype,
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs-of-iters",
        str(args.warmup_runs + args.test_runs),
    ]

    if not args.use_gpu:
        cmds.append("--use-cpu")

    if args.dry_run:
        print("Command to be executed:")
        print(" ".join(cmds))
        return 0

    p = subprocess.Popen(
        cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output = ""
    for line in iter(p.stdout.readline, ""):
        print(line, end="")
        if args.output:
            output += line

    p.stdout.close()
    p.wait()
    if args.output:
        with open(args.output, "w") as outf:
            outf.write(output)
    return p.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--python",
        type=str,
        default="python3.10",
        help="The python interpreter used to run the benchmark",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default="split_table_batched_embeddings_benchmark.py",
        help="Path to the benchmark script",
    )
    parser.add_argument(
        "--benchmark-cmd",
        type=str,
        default="nbit-device-with-spec",
        help="The subcommand of the benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--bag-size", type=int, default=13, help="Bag size or pooling factor"
    )
    parser.add_argument(
        "--shapes-file",
        type=str,
        required=True,
        help="Path to the JSON file that describes a list of shapes [rows, embedding-dims]. "
        + "Its content should look like '[[123, 2], [456, 16], [789, 16], ...]'",
    )
    parser.add_argument(
        "--weights-precision",
        type=str,
        default="fp16",
        help="Weight data type",
    )
    parser.add_argument(
        "--output-dtype", type=str, default="fp16", help="Output data type"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=5, help="Number of warmup runs"
    )
    parser.add_argument("--test-runs", type=int, default=5, help="Number of test runs")
    parser.add_argument(
        "--output", type=str, default="", help="Also log the benchmark output to a file"
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU instead of CPU")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print out the command that will execute",
    )
    args = parser.parse_args()
    returncode = run(args)
    exit(returncode)
