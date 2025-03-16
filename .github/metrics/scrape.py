#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta

import click

import github
from github import GitHubClient

@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--repo",
    default="pytorch/fbgemm",
    help="Repository in owner/repo format (default: pytorch/fbgemm).",
)
@click.option(
    "--last",
    default=7,
    type=int,
    help="Number of days to look back (e.g., 7 for the last 7 days).",
)
@click.option(
    "--branch", default="main", help="Target branch to filter by (default: main)."
)
@click.option(
    "--labels",
    help="Comma-separated list of labels to filter PRs by (e.g., bug,enhancement).",
)
def fetch(repo, last, branch, labels):
    """
    Fetches merged PRs in a given repository within a date range, filtered by target branch and labels.
    """
    # Calculate date range based on --last flag
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=last)

    # Convert dates to strings
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Parse labels
    labels = set(labels.split(",") if labels else [])

    # Fetch merged PRs
    merged_prs = [
        pr
        for pr in GitHubClient().fetch_closed_prs(repo, start_date_str, end_date_str, branch)
        if pr.merged()
    ]

    # Filter by labels
    filtered_prs = [
        pr for pr in merged_prs if ((pr.labels & labels) if labels else True)
    ]

    # Output results
    if filtered_prs:
        print(
            f"Found {len(filtered_prs)} merged PRs in '{repo}' between {start_date_str} and {end_date_str} into branch '{branch}':"
        )
        for pr in filtered_prs:
            print(pr.tostr())
    else:
        print(
            f"No merged PRs found in '{repo}' between {start_date_str} and {end_date_str} into branch '{branch}'."
        )


if __name__ == "__main__":
    cli()
