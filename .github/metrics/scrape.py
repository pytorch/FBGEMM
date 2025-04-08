# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta, timezone

import click

from reporters import ReleasesReporter, UnlabeledPrsReporter


def common_options(func):
    options = [
        click.option(
            "--owner",
            type=str,
            default="pytorch",
            help="Repository owner (default: pytorch).",
        ),
        click.option(
            "--repo",
            type=str,
            default="fbgemm",
            help="Repository name (default: fbgemm).",
        ),
        click.option(
            "--ref",
            type=str,
            default="main",
            help="Branch, tag, or SHA to filter by (default: main).",
        ),
        click.option(
            "--since",
            type=click.DateTime(formats=["%Y-%m-%d"]),
            default=datetime.now(timezone.utc) - timedelta(days=7),
            help="Start date in YYYY-MM-DD format (default: 7 days ago).",
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


@click.group()
def cli() -> None:
    pass


@cli.command()
@common_options
def release_report(owner: str, repo: str, ref: str, since: str | datetime):
    reporter = ReleasesReporter(owner, repo)
    reporter.print_report(ref, since)


@cli.command()
@common_options
def unlabeled_report(owner: str, repo: str, ref: str, since: str | datetime):
    reporter = UnlabeledPrsReporter(owner, repo)
    reporter.print_report(ref, since)


if __name__ == "__main__":
    cli()
