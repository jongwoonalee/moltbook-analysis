#!/bin/bash

# Unsubscribe from a submolt
# Usage: ./unsubscribe_submolt.sh SUBMOLT_NAME

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 SUBMOLT_NAME"
  exit 1
fi

SUBMOLT="$1"

curl -X DELETE "https://www.moltbook.com/api/v1/submolts/$SUBMOLT/subscribe" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
