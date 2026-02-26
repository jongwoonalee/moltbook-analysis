#!/bin/bash

# Get a single post by ID
# Usage: ./get_post.sh POST_ID

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 POST_ID"
  exit 1
fi

POST_ID="$1"

curl "https://www.moltbook.com/api/v1/posts/$POST_ID" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
