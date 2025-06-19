# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import datetime
from typing import List, Tuple

import requests

from datatypes import GHCommit, GHPullRequest
from dateutil import parser as dtparser  # For flexible date parsing
from ratelimit import limits, sleep_and_retry

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 20
logging.basicConfig(level=logging.INFO)


class GitHubClient:
    # GitHub API base URL
    API_URL = "https://api.github.com"

    def __init__(self, owner: str, repo: str) -> None:
        # Replace with your GitHub token and the repo details
        if "GITHUB_TOKEN" not in os.environ:
            raise Exception("GITHUB_TOKEN not set")
        self.token = os.environ["GITHUB_TOKEN"]

        self.owner = owner
        self.repo = repo

        # Headers for authentication
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def fetch_commits(self, ref: str, since_date: str | datetime) -> List[GHCommit]:
        """
        Fetch commits with PR numbers (for merge commits)

        Args:
            ref (str): Branch, tag, or SHA
            since_date (str): Date in YYYY-MM-DD format

        Returns:
            list: List of commits, sorted by date.
        """
        base_url = f"{self.API_URL}/repos/{self.owner}/{self.repo}/commits"

        since_str = (
            f"{since_date}T00:00:00Z"
            if isinstance(since_date, str)
            else since_date.isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        params = {"sha": ref, "since": since_str, "per_page": 100}
        logging.info(
            f"Fetching commits for {self.owner}/{self.repo}:{ref} since {since_str} ..."
        )

        commits = []
        page = 1

        while True:
            params["page"] = page
            response = requests.get(base_url, headers=self.headers, params=params)
            response.raise_for_status()

            current_commits = response.json()
            if not current_commits:
                break

            for commit in current_commits:
                commits.append(
                    GHCommit(
                        commit["sha"],
                        dtparser.parse(commit["commit"]["author"]["date"]),
                        commit["commit"]["message"],
                    )
                )

            page += 1

        return commits

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def pr_for_commit(self, commit_sha: str) -> GHPullRequest | None:
        """
        Fetch pull request information from GitHub API given the commit SHA.

        Args:
            pr_number: Pull request number

        Returns:
            GHPullRequest containing PR information, or None if no PR found

        Raises:
            requests.exceptions.HTTPError: If the request fails
        """
        logging.info(
            f"Fetching PR associated with {self.owner}/{self.repo}:{commit_sha} ..."
        )
        pr_id = self.pr_id_for_commit(commit_sha)
        return self.fetch_pr(pr_id) if pr_id else None

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def pr_id_for_commit(self, commit_sha: str) -> int | None:
        """
        Find PR number associated with a commit
        """
        search_url = f"{self.API_URL}/search/issues"

        query = f"repo:{self.owner}/{self.repo} is:pr {commit_sha}"
        params = {"q": query}

        response = requests.get(search_url, headers=self.headers, params=params)
        response.raise_for_status()

        results = response.json()
        if results["total_count"] > 0:
            return results["items"][0]["number"]
        return None

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def fetch_pr(self, pr_number: int) -> GHPullRequest:
        """
        Fetch pull request information from GitHub API.

        Args:
            pr_number: Pull request number

        Returns:
            GHPullRequest containing PR information

        Raises:
            requests.exceptions.HTTPError: If the request fails
        """
        url = f"{self.API_URL}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors

        # Get labels (requires separate API call)
        labels_url = (
            f"{self.API_URL}/repos/{self.owner}/{self.repo}/issues/{pr_number}/labels"
        )
        labels_response = requests.get(labels_url, headers=self.headers)
        labels = (
            [label["name"] for label in labels_response.json()]
            if labels_response.ok
            else []
        )

        pr_info = response.json()
        closed_at = pr_info.get("closed_at")
        closed_at = dtparser.parse(closed_at) if closed_at else None

        return GHPullRequest(
            pr_info["title"], pr_number, closed_at, set(labels), pr_info["base"]["ref"]
        ).cleaned()

    def fetch_prs(self, ref: str, since_date: str | datetime) -> List[GHPullRequest]:
        """
        Fetch PRs merged into a given branch within a date range.

        Args:
            ref (str): Branch, tag, or SHA
            since_date (str): Date in YYYY-MM-DD format

        Returns:
            list: List of PRs, sorted by date.
        """
        commits = self.fetch_commits(ref, since_date)
        prs = [self.pr_for_commit(c.sha) for c in commits]
        return [pr for pr in prs if pr is not None]

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
    def fetch_unlabeled_commits(
        self, ref: str, since_date: str | datetime
    ) -> List[Tuple[GHCommit, GHPullRequest]]:
        """
        Fetch commitswith no PR labels, along with their associated PRs.

        Args:
            ref (str): Branch, tag, or SHA
            since_date (str): Date in YYYY-MM-DD format

        Returns:
            List of (GHCommit, GHPullRequest) tuples, sorted by commit date.
        """
        commits = self.fetch_commits(ref, since_date)
        prs = [self.pr_for_commit(c.sha) for c in commits]
        return [
            (c, pr) for c, pr in zip(commits, prs) if pr is not None and pr.unlabeled()
        ]
