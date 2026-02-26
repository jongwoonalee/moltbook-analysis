#!/bin/bash

# Register a new agent on Moltbook
# Usage: ./register.sh "AgentName" "Agent description"

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 \"AgentName\" \"Description\""
  echo "Example: $0 \"MyAgent\" \"A helpful AI assistant\""
  exit 1
fi

AGENT_NAME="$1"
DESCRIPTION="$2"

curl -X POST https://www.moltbook.com/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"$AGENT_NAME\", \"description\": \"$DESCRIPTION\"}" | jq .
