# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set


@dataclass(frozen=True)
class GHCommit:
    """
    A commit on GitHub.
    """

    sha: str
    timestamp: datetime
    message: str


@dataclass
class GHPullRequest:
    """
    A pull request on GitHub.

    NOTE: Additional properties of the PR beyond those offered by GitHub API are
    stored as `key:value` string labels attached to the PR.
    """

    title: str
    number: int
    closed_at: Optional[datetime]
    labels: Set[str]
    base_ref: str

    def category(self) -> str:
        """
        The category of the PR, e.g. "new", "improvement", "fix".
        """
        tmp = [
            label.replace("category:", "")
            for label in self.labels
            if label.startswith("category:")
        ]
        return tmp[0] if len(tmp) > 0 else "misc"

    def feature(self) -> str:
        """
        The feature set that the PR belongs to.
        """
        tmp = [
            label.replace("feature:", "")
            for label in self.labels
            if label.startswith("feature:")
        ]
        return sorted(tmp)[0] if len(tmp) > 0 else "misc"

    def contributor(self) -> str:
        """
        The organization contributor of the PR, e.g. "Meta", "AMD".
        """
        tmp = [
            label.replace("contributor:", "")
            for label in self.labels
            if label.startswith("contributor:")
        ]
        return tmp[0] if len(tmp) > 0 else "Meta"

    def unlabeled(self):
        """
        Whether the PR is unlabeled
        """
        return len(self.labels) == 0

    def cleaned(self):
        """
        Remove the labels that are common to the PR.
        """
        misc = ["fb-exported", "cla signed", "ci-no-td", "Merged"]
        new_labels = [x for x in self.labels if x not in misc]
        return dataclasses.replace(self, labels=new_labels)

    def to_str(self):
        """
        A string representation of the PR
        """
        return f"[{self.category().title()}] {self.title} (#{self.number})"
