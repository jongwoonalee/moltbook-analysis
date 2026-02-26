#!/bin/bash

# Semantic search for posts and comments
# Usage: ./search.sh "query" [type] [limit]
# type: all, posts, comments (default: all)
# limit: max results (default: 20, max: 50)

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 \"query\" [type] [limit]"
  echo "Example: $0 \"how do agents handle memory\" posts 10"
  exit 1
fi

QUERY="$1"
TYPE="${2:-all}"
LIMIT="${3:-20}"

# URL encode the query
ENCODED_QUERY=$(echo "$QUERY" | jq -sRr @uri)

curl "https://www.moltbook.com/api/v1/search?q=$ENCODED_QUERY&type=$TYPE&limit=$LIMIT" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
