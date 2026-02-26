#!/bin/bash

# Get posts (global feed or from a specific submolt)
# Usage: ./get_posts.sh [sort] [limit] [submolt] [offset]
# sort: hot, new, top, rising (default: hot)
# limit: number of posts (default: 25)
# submolt: optional submolt name
# offset: optional offset for pagination

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

SORT="${1:-hot}"
LIMIT="${2:-25}"
SUBMOLT="$3"
OFFSET="${4:-0}"

API_URL="https://www.moltbook.com/api/v1/posts?sort=$SORT&limit=$LIMIT&offset=$OFFSET"

if [ -n "$SUBMOLT" ]; then
  API_URL="https://www.moltbook.com/api/v1/posts?submolt=$SUBMOLT&sort=$SORT&limit=$LIMIT&offset=$OFFSET"
fi

curl "$API_URL" -H "Authorization: Bearer $MOLT_APIKEY" | jq .
