#!/bin/bash

PORT=8000

# Find the process using the port
PID=$(lsof -ti :$PORT)

if [ -n "$PID" ]; then
    echo "Killing process on port $PORT (PID: $PID)..."
    kill -9 $PID
    echo "Port $PORT is now free."
else
    echo "No process found on port $PORT."
fi
