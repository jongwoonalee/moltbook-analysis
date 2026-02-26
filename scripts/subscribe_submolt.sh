#!/bin/bash

# Subscribe to a submolt
# Usage: ./subscribe_submolt.sh SUBMOLT_NAME

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 SUBMOLT_NAME"
  exit 1
fi

SUBMOLT="$1"

curl -X POST "https://www.moltbook.com/api/v1/submolts/$SUBMOLT/subscribe" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
