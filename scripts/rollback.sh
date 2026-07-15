#!/usr/bin/env bash
# Repoint the `live` alias at the previous published version. deploy.yml does this automatically
# when its post-deploy smoke fails; this is the manual path for a problem found later.
set -euo pipefail

STACK=${STACK:-arxiv-atlas}
REGION=${AWS_REGION:-us-east-1}

fn=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue" --output text)
current=$(aws lambda get-alias --function-name "$fn" --name live --region "$REGION" \
  --query FunctionVersion --output text)

# Published versions are monotonic integers, so the rollback target is the highest one below the
# alias's current version. $LATEST is excluded: it tracks the newest code, which is what we are
# rolling away from.
target=$(aws lambda list-versions-by-function --function-name "$fn" --region "$REGION" \
  --query "Versions[?Version!='\$LATEST'].Version" --output text \
  | tr '\t' '\n' | sort -n | awk -v cur="$current" '$1 + 0 < cur + 0' | tail -1)

if [ -z "$target" ]; then
  echo "no published version below $current to roll back to" >&2
  exit 1
fi

echo "rolling live back: $current -> $target"
aws lambda update-alias --function-name "$fn" --name live --function-version "$target" \
  --region "$REGION" --query FunctionVersion --output text
echo "done. The stack drifts from the alias until the next deploy re-shifts it."
