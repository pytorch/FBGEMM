#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import subprocess
from datetime import datetime
from typing import Union

import click

import pandas as pd

import torch


class CLIOutput:
    def __init__(
        self,
        cli: str = "",
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        timestamp: str = "2025-01-01T20:00:00.00000",
        visible: bool = True,
    ) -> None:
        self._cli = cli
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._timestamp = timestamp
        self._visible = visible

    def to_dict(self) -> dict[str, Union[int, str]]:
        return {
            "cli": self._cli,
            "stdout": self._stdout,
            "stderr": self._stderr,
            "returncode": self._returncode,
            "timestamp": self._timestamp,
            "visible": self._visible,
        }


class CLI:
    def __init__(self) -> None:
        pd.options.display.max_rows
        pd.set_option("display.max_colwidth", None)
        self._cli_outputs: list[CLIOutput] = [
            CLIOutput(
                cli="python –c “import torch; print(torch.__version__)”",
                stdout="{}".format(torch.__version__),
                stderr="",
                returncode=0,
                timestamp=datetime.now().isoformat(),
                visible=True,
            )
        ]

    def run(
        self,
        cli: Union[str, list[str]],
        visible: bool = True,
        input: str = "",
        capture_output: bool = True,
    ) -> CLIOutput:
        if isinstance(cli, str):
            cli = cli.split()
        result = CLIOutput()
        try:
            completed = subprocess.run(
                cli, text=True, check=False, capture_output=capture_output, input=input
            )
            result = CLIOutput(
                cli=" ".join(cli),
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                timestamp=datetime.now().isoformat(),
                visible=visible,
            )
            if visible:
                self._cli_outputs.append(result)
        except Exception as e:
            logging.error(f'For cli {" ".join(cli)} we got exception {e}')
            result = CLIOutput(
                cli=" ".join(cli),
                stdout="",
                stderr=str(e),
                returncode=-1,
                visible=visible,
                timestamp=datetime.now().isoformat(),
            )
            if visible:
                self._cli_outputs.append(result)
        return result

    def run_piped(self, clis: list[str]) -> None:
        the_input = ""
        for cli in clis[:-1]:
            result = self.run(
                cli=cli, visible=False, input=the_input, capture_output=True
            )
            the_input = result._stdout
        self.run(cli=clis[-1], visible=True, input=the_input, capture_output=True)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([output.to_dict() for output in self._cli_outputs])

    def save(self, filename: str, format: str = "csv") -> None:
        df = self.to_dataframe()
        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "json":
            df.to_json(filename, orient="records", lines=True)
        else:
            raise ValueError(f"Invalid format {format} : must be one of 'csv', 'json'")


@click.command()
@click.option("--json", default="")
@click.option("--csv", default="")
def cli_run(
    json: str,
    csv: str,
) -> None:
    cli = CLI()

    the_rpm = "rpm -qa"
    the_grep1 = "grep -E ^amdgpu-(dkms|kmod)"
    the_grep2 = "grep -v firmware"
    the_sed1 = "sed -E s/^[^-]-[^-]-//"
    the_sed2 = "sed -E s/.[^.].[^.]$//"
    cli.run_piped([the_rpm, the_grep1, the_grep2, the_sed1, the_sed2])

    cli.run("uname -r")

    cli.run("fw-util all --version")

    cli.run("amd-smi firmware")
    cli.run("amd-smi version")
    cli.run("amd-smi static")

    if len(csv):
        cli.save(csv)

    if len(json):
        cli.save(json, format="json")

    print(cli.to_dataframe())


def main() -> None:
    cli_run()


if __name__ == "__main__":
    main()
