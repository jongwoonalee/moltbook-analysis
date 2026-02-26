#!/bin/bash

# List all submolts
# Usage: ./list_submolts.sh

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

curl https://www.moltbook.com/api/v1/submolts&limit=100 \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
