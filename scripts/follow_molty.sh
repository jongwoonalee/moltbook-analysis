#!/bin/bash

# Follow a molty (another agent)
# Usage: ./follow_molty.sh MOLTY_NAME

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 MOLTY_NAME"
  exit 1
fi

MOLTY="$1"

curl -X POST "https://www.moltbook.com/api/v1/agents/$MOLTY/follow" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
