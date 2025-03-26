#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Set
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


class GitHubClient:
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



    # def get_commits_since_sha(self, repo, branch=None, since_sha=None):
    #     """Fetch all commits since a specific date or SHA, optionally filtered by branch."""
    #     commits_url = f"{self.API_URL}/{repo}/commits"
    #     params = {
    #         'per_page': 100,  # Max number of commits per page
    #         'sha': branch
    #     }

    #     commits = []
    #     while True:
    #         response = requests.get(commits_url, headers=self.headers, params=params)
    #         response.raise_for_status()
    #         new_commits = response.json()
    #         if since_sha:
    #             # Stop fetching if we reach the since_sha commit
    #             if any(commit['sha'] == since_sha for commit in new_commits):
    #                 break
    #         commits.extend(new_commits)
    #         if 'next' in response.links:
    #             commits_url = response.links['next']['url']
    #         else:
    #             break
    #     return commits
