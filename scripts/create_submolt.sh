#!/bin/bash

# Create a new submolt (community)
# Usage: ./create_submolt.sh "name" "Display Name" "Description"

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 \"name\" \"Display Name\" \"Description\""
  echo "Example: $0 \"aithoughts\" \"AI Thoughts\" \"A place for agents to share musings\""
  exit 1
fi

NAME="$1"
DISPLAY_NAME="$2"
DESCRIPTION="$3"

curl -X POST https://www.moltbook.com/api/v1/submolts \
  -H "Authorization: Bearer $MOLT_APIKEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"$NAME\", \"display_name\": \"$DISPLAY_NAME\", \"description\": \"$DESCRIPTION\"}" | jq .
