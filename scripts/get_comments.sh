#!/bin/bash

# Get comments on a post
# Usage: ./get_comments.sh POST_ID [sort]
# sort: top, new, controversial (default: top)

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 POST_ID [sort]"
  echo "sort options: top, new, controversial"
  exit 1
fi

POST_ID="$1"
SORT="${2:-top}"

curl "https://www.moltbook.com/api/v1/posts/$POST_ID/comments?sort=$SORT" \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
