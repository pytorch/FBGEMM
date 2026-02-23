#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Trigger PyTorch binaries build by cloning at a specific SHA, creating a PR,
# and adding the ciflow/binaries label. Requires: gh CLI, fork of pytorch/pytorch.

set -eo pipefail

PYTORCH_REPO="pytorch/pytorch"
PYTORCH_URL="https://github.com/pytorch/pytorch.git"
CIFLOW_LABEL="ciflow/binaries"
GH_BIN="${GH_PATH:-gh}"

info() { echo "[INFO] $*"; }
err()  { echo "[ERROR] $*" >&2; }
step() { echo -e "\n=== $* ==="; }

usage() {
    cat << 'EOF'
Usage: ./pytorch_trigger_binaries_build.bash <GIT_SHA> [OPTIONS]

Trigger a PyTorch binaries build by creating a PR with the ciflow/binaries label.

Arguments:
  GIT_SHA              The PyTorch commit SHA to build binaries for

Options:
  --fork <username>    GitHub username for the fork (default: auto-detect)
  --clone-dir <dir>    Directory to clone PyTorch into (default: ~/pytorch-<sha>)
  --branch <name>      Branch name to create (default: binaries-build-<sha>-<timestamp>)
  --dry-run            Show what would be done without making changes
  --help               Show this help message

Environment:
  GH_PATH              Path to gh executable (default: gh in PATH)

Examples:
  ./pytorch_trigger_binaries_build.bash abc123def456
  ./pytorch_trigger_binaries_build.bash abc123def456 --fork myusername --dry-run
EOF
}

check_prereqs() {
    step "Checking prerequisites"
    command -v "$GH_BIN" &>/dev/null || { err "gh CLI not found. Install: https://cli.github.com/"; return 1; }
    "$GH_BIN" auth status &>/dev/null || { err "gh not authenticated. Run: $GH_BIN auth login"; return 1; }
    command -v git &>/dev/null || { err "git not installed"; return 1; }
    info "gh: $("$GH_BIN" --version | head -1), git: $(git --version)"
}

get_gh_user() {
    "$GH_BIN" api user --jq '.login' 2>/dev/null || echo "";
}

get_gh_name() {
    "$GH_BIN" api user --jq '.name // .login' 2>/dev/null || echo "";
}

get_gh_email() {
    local user email
    user=$(get_gh_user)
    [[ -z "$user" ]] && return 1

    # Try primary email from GitHub API (requires user:email scope)
    email=$("$GH_BIN" api user/emails --jq '[.[] | select(.primary==true)][0].email' 2>/dev/null)

    # Validate email format; fallback to noreply email (always valid for EasyCLA)
    if [[ "$email" =~ ^[^@]+@[^@]+\.[^@]+$ && "$email" != *"{"* ]]; then
        echo "$email"
    else
        echo "${user}@users.noreply.github.com"
    fi
}

verify_fork() {
    "$GH_BIN" repo view "$1/pytorch" &>/dev/null || { err "Fork not found: $1/pytorch"; return 1; }
    info "Fork verified: $1/pytorch"
}

main() {
    local git_sha="" fork_owner="" clone_dir="" branch_name="" dry_run=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fork)      fork_owner="$2"; shift 2 ;;
            --clone-dir) clone_dir="$2"; shift 2 ;;
            --branch)    branch_name="$2"; shift 2 ;;
            --dry-run)   dry_run=true; shift ;;
            --help|-h)   usage; exit 0 ;;
            -*)          err "Unknown option: $1"; usage; exit 1 ;;
            *)           if [[ -z "$git_sha" ]]; then git_sha="$1"; else err "Unexpected: $1"; exit 1; fi; shift ;;
        esac
    done

    [[ -z "$git_sha" ]] && { err "Missing GIT_SHA"; usage; exit 1; }
    : "${clone_dir:=${HOME}/pytorch-${git_sha:0:12}}"
    : "${branch_name:=binaries-build-${git_sha:0:8}-$(date +%Y%m%d-%H%M%S)}"

    echo "=== PyTorch Binaries Build Trigger ==="
    check_prereqs || exit 1

    if [[ -z "$fork_owner" ]]; then
        fork_owner=$(get_gh_user)
        [[ -z "$fork_owner" ]] && { err "Could not detect GitHub username. Use --fork"; exit 1; }
    fi
    verify_fork "$fork_owner" || exit 1

    info "SHA: $git_sha | Fork: $fork_owner/pytorch | Branch: $branch_name"
    [[ "$dry_run" == true ]] && { info "DRY RUN MODE - no changes will be made"; }

    # Sync fork with upstream
    step "Syncing fork with upstream"
    if [[ "$dry_run" == true ]]; then
        info "[DRY RUN] Would sync $fork_owner/pytorch with $PYTORCH_REPO"
    else
        "$GH_BIN" repo sync "$fork_owner/pytorch" --source "$PYTORCH_REPO" --branch main
        info "Fork synced"
    fi

    # Clone and checkout
    step "Cloning repository"
    if [[ "$dry_run" == true ]]; then
        info "[DRY RUN] Would clone $PYTORCH_URL to $clone_dir and checkout $git_sha"
    else
        [[ -d "$clone_dir" ]] && rm -rf "$clone_dir"
        git clone --depth=1 "$PYTORCH_URL" "$clone_dir"
        cd "$clone_dir"
        git fetch --depth=1 origin "$git_sha"
        git checkout "$git_sha"
    fi

    # Create branch and commit
    step "Creating branch"
    if [[ "$dry_run" == true ]]; then
        info "[DRY RUN] Would create branch $branch_name with trigger commit"
    else
        local gh_name gh_email
        gh_name=$(get_gh_name)
        gh_email=$(get_gh_email)
        [[ -z "$gh_name" || -z "$gh_email" ]] && { err "Could not fetch GitHub user info"; exit 1; }

        git config user.name "$gh_name"
        git config user.email "$gh_email"
        info "Git configured: $gh_name <$gh_email>"

        git checkout -b "$branch_name"
        git remote add fork "https://github.com/${fork_owner}/pytorch.git"
        git commit --allow-empty -m "Trigger binaries build for ${git_sha:0:8}

This PR triggers the PyTorch binaries CI workflow for commit ${git_sha}.
Requested by: ${fork_owner} | Target SHA: ${git_sha}"
    fi

    # Push and create PR
    step "Pushing and creating PR"
    local pr_url=""
    if [[ "$dry_run" == true ]]; then
        info "[DRY RUN] Would push branch and create PR with label: $CIFLOW_LABEL"
    else
        git push -u fork "$branch_name"
        pr_url=$("$GH_BIN" pr create --repo "$PYTORCH_REPO" \
            --head "${fork_owner}:${branch_name}" --base main --draft \
            --title "[CI] Trigger binaries build for ${git_sha:0:8}" \
            --body "Triggers binaries CI for \`${git_sha}\`. Close after CI completes (do not merge).")
        info "PR created: $pr_url"

        "$GH_BIN" pr edit "${pr_url##*/}" --repo "$PYTORCH_REPO" --add-label "$CIFLOW_LABEL"
        info "Label '$CIFLOW_LABEL' added"
    fi

    echo -e "\n=== SUCCESS ==="
    if [[ "$dry_run" == true ]]; then
        echo "Dry run complete. No changes made."
    else
        echo "PR: $pr_url"
    fi
    echo "Clone dir: $clone_dir"
}

main "$@"
