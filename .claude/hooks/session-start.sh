#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs the Google Cloud SDK and the repo's Python dev dependencies, and
# authenticates to GCP if a service account key is available, so future agent
# sessions can read from gs://every-query-runs and run the test / lint suite.
#
# Only runs in remote (web) sessions. Local sessions are a no-op.

set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    echo "[session-start] not a remote session, skipping."
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
GCLOUD_HOME="$HOME/google-cloud-sdk"

# ---------------------------------------------------------------------------
# 1. Install the Google Cloud SDK (gcloud + gsutil) if missing.
# ---------------------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1 && [ ! -x "$GCLOUD_HOME/bin/gcloud" ]; then
    echo "[session-start] installing Google Cloud SDK into $GCLOUD_HOME"
    TMP_TARBALL="$(mktemp --suffix=.tar.gz)"
    curl -fsSL -o "$TMP_TARBALL" \
        https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xzf "$TMP_TARBALL" -C "$HOME"
    rm -f "$TMP_TARBALL"
    "$GCLOUD_HOME/install.sh" --quiet --path-update=false --usage-reporting=false \
        --command-completion=false >/dev/null
else
    echo "[session-start] Google Cloud SDK already installed."
fi

# Put gcloud/gsutil on PATH for this session and all future tool invocations.
export PATH="$GCLOUD_HOME/bin:$PATH"
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo "export PATH=\"$GCLOUD_HOME/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 2. Authenticate to GCP if a service account key was provided.
#
# Provide the key via one of (checked in order):
#   * GCP_SERVICE_ACCOUNT_KEY       — full JSON contents of the key file
#   * GOOGLE_APPLICATION_CREDENTIALS — path to a mounted key file
#
# Configure either as a session secret in the Claude Code web UI.
# ---------------------------------------------------------------------------
KEY_FILE="$HOME/.config/gcloud-service-account.json"
mkdir -p "$(dirname "$KEY_FILE")"

if [ -n "${GCP_SERVICE_ACCOUNT_KEY:-}" ]; then
    printf '%s' "$GCP_SERVICE_ACCOUNT_KEY" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
fi

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "[session-start] activating service account from $GOOGLE_APPLICATION_CREDENTIALS"
    gcloud auth activate-service-account \
        --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet
    if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
        echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$GOOGLE_APPLICATION_CREDENTIALS\"" \
            >> "$CLAUDE_ENV_FILE"
    fi
else
    echo "[session-start] no GCP_SERVICE_ACCOUNT_KEY / GOOGLE_APPLICATION_CREDENTIALS found;"
    echo "[session-start] gcloud is installed but unauthenticated. Set the secret in the"
    echo "[session-start] Claude Code web UI to enable gs://every-query-runs access."
fi

# ---------------------------------------------------------------------------
# 3. Install Python dev dependencies so tests and linters work.
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "[session-start] syncing Python deps with uv"
    (cd "$PROJECT_DIR" && uv sync --locked --group dev)
else
    echo "[session-start] uv not found on PATH; skipping Python dep install."
fi

echo "[session-start] done."
