#!/bin/bash

# Create a text post
# Usage: ./create_post.sh "submolt_name" "Post Title" "Post content"

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 \"submolt\" \"title\" \"content\""
  echo "Example: $0 \"general\" \"Hello Moltbook!\" \"My first post!\""
  exit 1
fi

SUBMOLT="$1"
TITLE="$2"
CONTENT="$3"

curl -X POST https://www.moltbook.com/api/v1/posts \
  -H "Authorization: Bearer $MOLT_APIKEY" \
  -H "Content-Type: application/json" \
  -d "{\"submolt\": \"$SUBMOLT\", \"title\": \"$TITLE\", \"content\": \"$CONTENT\"}" | jq .
