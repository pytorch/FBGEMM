# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from datetime import datetime

from github import GitHubClient


class ReleasesReporter:
    def __init__(self, owner: str, repo: str) -> None:
        self.client = GitHubClient(owner, repo)

    def print_report(self, ref: str, since_date: str | datetime) -> None:
        pull_requests = self.client.fetch_prs(ref, since_date)
        buckets = itertools.groupby(
            sorted(pull_requests, key=lambda x: x.feature()), lambda x: x.feature()
        )

        logging.info("Generating report ...\n\n\n")

        for feature, group in buckets:
            print(f"# {feature.upper()}")
            for pr in group:
                print(f"- {pr.to_str()}")
            print("")


class UnlabeledPrsReporter:
    def __init__(self, owner: str, repo: str) -> None:
        self.owner = owner
        self.repo = repo
        self.client = GitHubClient(owner, repo)

    def print_report(self, ref: str, since_date: str | datetime) -> None:
        tuples = self.client.fetch_unlabeled_commits(ref, since_date)

        logging.info("Generating report ...\n\n\n")

        for c, pr in tuples:
            print(
                f"{c.timestamp} {c.sha} https://github.com/{self.owner}/{self.repo}/pull/{pr.number}"
            )
