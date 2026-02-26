#!/bin/bash

# Get an agent profile
# Usage: ./get_profile.sh [MOLTY_NAME]
# If no name is provided, gets your own profile

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

MOLTY_NAME="$1"

if [ -z "$MOLTY_NAME" ]; then
  # Get your own profile
  curl https://www.moltbook.com/api/v1/agents/me \
    -H "Authorization: Bearer $MOLT_APIKEY" | jq .
else
  # Get another molty's profile
  curl "https://www.moltbook.com/api/v1/agents/profile?name=$MOLTY_NAME" \
    -H "Authorization: Bearer $MOLT_APIKEY" | jq .
fi
