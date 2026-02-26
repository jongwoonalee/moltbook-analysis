#!/bin/bash

# Downvote a post
# Usage: ./downvote_post.sh POST_ID

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 POST_ID"
  exit 1
fi

POST_ID="$1"

curl -X POST "https://www.moltbook.com/api/v1/posts/$POST_ID/downvote" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
