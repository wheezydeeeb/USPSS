#!/bin/bash

# Define the Python file name
python_file="csv_log_face.py"

# Check if the file exists
if [ ! -f "$python_file" ]; then
    echo "Error: File '$python_file' not found."
    exit 1
fi

# Run the Python file using Python interpreter
python3 "$python_file"

