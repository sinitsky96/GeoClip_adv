#!/bin/bash

# Default to download all rows if MAX_ROWS is not set
MAX_ROWS=${MAX_ROWS:-0}

# Check if file already exists in its final form
if [ -f ./mp16_urls.csv ]; then
    echo "MP-16 URLs CSV file already exists, skipping download"
    exit 0
fi

# Download the MP-16 URLs CSV file
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv -O ./mp16_urls_full.csv

# Check if we need to limit the rows
if [ "$MAX_ROWS" -gt 0 ]; then
    echo "Limiting to $MAX_ROWS rows (plus header)"
    # Keep the header and the specified number of rows
    head -n 1 ./mp16_urls_full.csv > ./mp16_urls.csv
    tail -n +2 ./mp16_urls_full.csv | head -n $MAX_ROWS >> ./mp16_urls.csv
    echo "Created limited CSV file with $(expr $MAX_ROWS + 1) rows (including header)"
    # Remove the full file to save space
    rm ./mp16_urls_full.csv
else
    # Use the full file
    mv ./mp16_urls_full.csv ./mp16_urls.csv
    echo "Using complete CSV file"
fi

echo "MP-16 URLs CSV file is ready"
echo "The actual images will be downloaded by the Python script when needed"