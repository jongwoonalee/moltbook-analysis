#!/bin/bash

# Create a link post
# Usage: ./create_link_post.sh "submolt_name" "Post Title" "https://url"

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 \"submolt\" \"title\" \"url\""
  echo "Example: $0 \"general\" \"Interesting article\" \"https://example.com\""
  exit 1
fi

SUBMOLT="$1"
TITLE="$2"
URL="$3"

curl -X POST https://www.moltbook.com/api/v1/posts \
  -H "Authorization: Bearer $MOLT_APIKEY" \
  -H "Content-Type: application/json" \
  -d "{\"submolt\": \"$SUBMOLT\", \"title\": \"$TITLE\", \"url\": \"$URL\"}" | jq .
