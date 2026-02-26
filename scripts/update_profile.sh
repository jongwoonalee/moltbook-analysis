#!/bin/bash

# Update your profile description
# Usage: ./update_profile.sh "New description"

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: $0 \"New description\""
  exit 1
fi

DESCRIPTION="$1"

curl -X PATCH https://www.moltbook.com/api/v1/agents/me \
  -H "Authorization: Bearer $MOLT_APIKEY" \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"$DESCRIPTION\"}" | jq .
