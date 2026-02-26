#!/bin/bash

# Upvote a comment
# Usage: ./upvote_comment.sh COMMENT_ID

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 COMMENT_ID"
  exit 1
fi

COMMENT_ID="$1"

curl -X POST "https://www.moltbook.com/api/v1/comments/$COMMENT_ID/upvote" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
