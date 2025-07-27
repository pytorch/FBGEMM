#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import subprocess
import pandas as pd
from datetime import datetime 
from typing import List, Union

class CLIOutput: 
    def __init__(self, cli: str, stdout: str, stderr: str, returncode: int, timestamp: str):
        self._cli = cli
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._timestamp = timestamp

    def to_dict(self): 
        return {
            "cli": self._cli,
            "stdout": self._stdout,
            "stderr": self._stderr,
            "returncode": self._returncode,
            "timestamp": self._timestamp,
        }

class CLI: 
    def __init__(self):
        self._cli_outputs: List[CLIOutput] = []

    def run(self, cli: Union[str, List[str]]):
        if isinstance(cli, str):
            cli = cli.split()
        
        try:
            completed = subprocess.run(cli, capture_output=True, text=True, check=False)
            result = CLIOutput(
                cli = ' '.join(cli),
                stdout = completed.stdout,
                stderr = completed.stderr,
                returncode = completed.stderr,
                timestamp = datetime.now().isoformat(),
            )
            self._cli_outputs.append(result)
        except Exception as e:
            self.results.append(CLIOuput(
                command = ' '.join(cli),
                stdout = '',
                stderr = str(e),
                returncode = -1,
                timestamp = datetime.now().isoformat(),
            ))
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([output.to_dict() for output in self._cli_outputs])

    def save(self, filename: str, format: str = "csv"):
        df = self.to_dataframe()
        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "json":
            df.to_json(filename, orient="records", lines=True)
        elif format == "parquet":
            df.to_parquet(filename, index=False)
        else:
            raise ValueError(f"Invalid format {format} : must be one of 'csv', 'json', or 'parquet'.")

if __name__ == "__main__":
    cli = CLI()
    cli.run("ls")
    cli.run("ls -l")
    cli.run("ls -l /tmp")
    cli.save("cli_run.csv")
    cli.save("cli_run.json", format="json")
    cli.save("cli_run.parquet", format="parquet")
    print(cli.to_dataframe())
