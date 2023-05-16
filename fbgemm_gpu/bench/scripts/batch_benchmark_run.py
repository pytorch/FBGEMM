#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import subprocess

import click

logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option(
    "--benchmark-command",
    default="python split_table_batched_embeddings_benchmark.py",
    help="Benchmark command to run",
)
@click.option(
    "--command-file",
    default="batch_input.txt",
    help="File containing input commands to evaluate",
)
def batch_benchmark(
    benchmark_command: str,
    command_file: str,
) -> None:
    assert (
        "split_table_batched_embeddings_benchmark" in benchmark_command
    ), "split_table_batched_embeddings benchmark required for execution"

    benchmark_cmd = benchmark_command.strip().split()

    cmds_run = 0
    failed_runs = []
    total_fwd_bytes_read_gb = 0
    total_fwdbwd_bytes_read_gb = 0
    total_fwd_time_us = 0
    total_fwdbwd_time_us = 0
    with open(command_file) as cmd_file:
        for line in cmd_file:
            options = line.replace('"', "").strip().split()
            cmd = benchmark_cmd + options
            logging.info(f"Running command {cmds_run}: {cmd}")
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            logging.info(result.stdout.decode("utf-8"))
            # Parse results
            found_fwd_results = False
            found_fwdbwd_results = False
            for line in result.stdout.decode("utf-8").splitlines():
                re_match = re.search(r"BW:  ([\.\d]+) GB/s, T: ([\.\d]+)us", line)
                if re_match:
                    bw_gb = float(re_match.groups()[0])
                    time_us = int(re_match.groups()[1])
                    total_bytes_read_gb = bw_gb * time_us / 1e6

                    if "Forward, " in line:
                        total_fwd_bytes_read_gb += total_bytes_read_gb
                        total_fwd_time_us += time_us
                        found_fwd_results = True
                    elif "ForwardBackward, " in line:
                        total_fwdbwd_bytes_read_gb += total_bytes_read_gb
                        total_fwdbwd_time_us += time_us
                        found_fwdbwd_results = True
                    else:
                        raise Exception(
                            f"Unexpected reported metric for line: '{line}'"
                        )
            if not (found_fwd_results and found_fwdbwd_results):
                failed_runs.append(cmds_run)
            cmds_run += 1
    logging.info(f"Number of commands run: {cmds_run}")
    if failed_runs:
        logging.info(f"Failed runs: {failed_runs}")
    logging.info(
        f"Average FWD BW: {total_fwd_bytes_read_gb / total_fwd_time_us * 1e6} GB/s"
    )
    logging.info(
        f"        FWDBWD BW: {total_fwdbwd_bytes_read_gb / total_fwdbwd_time_us * 1e6} GB/s"
    )


if __name__ == "__main__":
    batch_benchmark()
