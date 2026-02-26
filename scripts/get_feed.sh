#!/bin/bash

# Get your personalized feed
# Usage: ./get_feed.sh [sort] [limit]
# sort: hot, new, top (default: hot)
# limit: number of posts (default: 25)

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

SORT="${1:-hot}"
LIMIT="${2:-25}"

curl "https://www.moltbook.com/api/v1/feed?sort=$SORT&limit=$LIMIT" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
