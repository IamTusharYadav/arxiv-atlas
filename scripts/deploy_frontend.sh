#!/usr/bin/env bash
# Build the frontend and publish it to the Caddy box. The API URL is compiled into the bundle
# (Vite inlines import.meta.env at build time), so it is a build input, not a runtime setting.
#
#   SSH_TARGET=user@tusharyadav.dev scripts/deploy_frontend.sh
set -euo pipefail

: "${SSH_TARGET:?set SSH_TARGET, e.g. user@tusharyadav.dev}"
TARGET_DIR=${TARGET_DIR:-/var/www/atlas}
API_URL=${VITE_API_URL:-https://99a0zbyk70.execute-api.us-east-1.amazonaws.com}

cd "$(dirname "$0")/.."

echo "building against $API_URL"
VITE_API_URL="$API_URL" npm --prefix frontend ci
VITE_API_URL="$API_URL" npm --prefix frontend run build

# --delete keeps old asset hashes from piling up. Safe because index.html is served no-cache,
# so a client picks up the new asset names on its next load rather than requesting deleted ones.
rsync -avz --delete frontend/dist/ "$SSH_TARGET:$TARGET_DIR/"

echo "deployed. https://atlas.tusharyadav.dev"
