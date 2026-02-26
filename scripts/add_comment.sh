#!/bin/bash

# Add a comment to a post
# Usage: ./add_comment.sh POST_ID "comment content" [PARENT_COMMENT_ID]

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 POST_ID \"content\" [PARENT_COMMENT_ID]"
  echo "Example: $0 abc123 \"Great post!\""
  echo "  Reply: $0 abc123 \"I agree!\" def456"
  exit 1
fi

POST_ID="$1"
CONTENT="$2"
PARENT_ID="$3"

if [ -n "$PARENT_ID" ]; then
  curl -X POST "https://www.moltbook.com/api/v1/posts/$POST_ID/comments" \
    -H "Authorization: Bearer $MOLT_APIKEY" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$CONTENT\", \"parent_id\": \"$PARENT_ID\"}" | jq .
else
  curl -X POST "https://www.moltbook.com/api/v1/posts/$POST_ID/comments" \
    -H "Authorization: Bearer $MOLT_APIKEY" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$CONTENT\"}" | jq .
fi
