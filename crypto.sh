#!/bin/bash

while true; do
  # Start the python script in the background
  python crypto.py -t Minute -n 1 &
  
  # Get the PID of the last background command
  PID=$!

  # Wait for 60 seconds
  sleep 60

  # Kill the python script
  kill $PID

  # Wait a couple seconds to allow the process to terminate
  sleep 2
done
