#!/bin/bash

# Script to append results folder to results.zip
# Creates results.zip if it doesn't exist, otherwise appends to it

set -e  # Exit on any error

# Get the project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
ZIP_FILE="$PROJECT_ROOT/results.zip"

echo "=== Results Zipper (Append Mode) ==="
echo "Project root: $PROJECT_ROOT"
echo "Results directory: $RESULTS_DIR"
echo "Zip file: $ZIP_FILE"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory $RESULTS_DIR does not exist!"
    exit 1
fi

# Check if zip file exists
if [ -f "$ZIP_FILE" ]; then
    echo "✓ Found existing $ZIP_FILE, will append to it"
    ZIP_MODE="-u"  # Update mode (append)
else
    echo "✓ No existing zip file found, will create new one"
    ZIP_MODE=""     # Create new
fi

# Create or append to zip file
echo "Processing results directory..."
cd "$RESULTS_DIR"

# Use zip with update mode to append new files
if [ -n "$ZIP_MODE" ]; then
    # Append mode - add new files and update existing ones
    zip -r "$ZIP_MODE" "$ZIP_FILE" . -x "*.tmp" "*.log" "*/temp/*" "*/cache/*"
else
    # Create new zip
    zip -r "$ZIP_FILE" . -x "*.tmp" "*.log" "*/temp/*" "*/cache/*"
fi

# Check if zip was successful
if [ $? -eq 0 ]; then
    echo "✓ Successfully processed results directory"
    
    # Get file size
    ZIP_SIZE=$(stat -c%s "$ZIP_FILE" 2>/dev/null || stat -f%z "$ZIP_FILE" 2>/dev/null || echo "unknown")
    if [ "$ZIP_SIZE" != "unknown" ]; then
        ZIP_SIZE_MB=$(echo "scale=2; $ZIP_SIZE / 1024 / 1024" | bc 2>/dev/null || echo "unknown")
        echo "✓ Zip file size: ${ZIP_SIZE_MB}MB"
    fi
    
    echo "✓ Results folder processed successfully!"
else
    echo "✗ Failed to process results folder!"
    exit 1
fi
