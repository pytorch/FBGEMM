#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Set

import click
import requests


@dataclass
class PullRequestInfo:
    title: str
    number: int
    closed_at: datetime
    labels: Set[str]
    base_ref: str

    def cleaned(self):
        new_labels = [x for x in self.labels if x not in ["cla signed"]]
        return dataclasses.replace(self, labels=new_labels)

    def merged(self):
        return "Merged" in self.labels

    def tostr(self):
        return f"{self.closed_at} {self.title} (#{self.number})"


class GitHub:
    # GitHub API base URL
    API_URL = "https://api.github.com/repos"

    def __init__(self) -> None:
        # Replace with your GitHub token and the repo details
        if "GITHUB_TOKEN" not in os.environ:
            raise Exception("GITHUB_TOKEN not set")
        self.token = os.environ["GITHUB_TOKEN"]

        # Headers for authentication
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def fetch_closed_prs(self, repo, start_date, end_date, target_branch):
        """
        Fetches merged PRs within a specified date range that were merged into a given branch.

        Args:

            repo:           Repository in 'owner/repo' format.
            start_date:     Start date in 'YYYY-MM-DD' format.
            end_date:       End date in 'YYYY-MM-DD' format.
            target_branch:  The target branch PRs were merged into.

        Returns:
            List of merged PRs with details.
        """

        # Convert dates to datetime objects for comparison
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Endpoint to fetch closed PRs
        url = f"{self.API_URL}/{repo}/pulls?state=closed&per_page=100&sort=updated&direction=desc"
        closed_prs = []

        while url:
            print(url)
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break

            prs = response.json()
            for pr in prs:
                if pr["closed_at"] is not None:  # Check if the PR was closed
                    closed_at = datetime.strptime(pr["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
                    # Check if the merged date is within the specified range and the target branch matches
                    if (
                        start_date <= closed_at <= end_date
                        and pr["base"]["ref"] == target_branch
                    ):
                        # Fetch labels for the PR
                        labels = [label["name"] for label in pr["labels"]]
                        closed_prs.append(
                            PullRequestInfo(
                                pr["title"],
                                int(pr["number"]),
                                closed_at,
                                set(labels),
                                pr["base"]["ref"],
                            )
                        )

            # Check for pagination
            if "next" in response.links:
                url = response.links["next"]["url"]
            else:
                url = None

        return sorted(closed_prs, key=lambda x: x.closed_at, reverse=True)


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
        for pr in GitHub().fetch_closed_prs(repo, start_date_str, end_date_str, branch)
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
