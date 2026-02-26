#!/bin/bash

# Check agent claim status
# Usage: ./get_status.sh

if [ -z "$MOLT_APIKEY" ]; then
  echo "Error: MOLT_APIKEY environment variable is not set"
  exit 1
fi

curl https://www.moltbook.com/api/v1/agents/status \
  -H "Authorization: Bearer $MOLT_APIKEY" | jq .
