#!/bin/bash

# Download MOMA cell dataset from Zenodo
echo "Checking for existing dataset..."

# Check if CTC folder exists
if [ -d "CTC" ]; then
    echo "CTC folder already exists, skipping download..."
else
    # Download dataset using curl with progress
    echo "Downloading CTC.zip from Zenodo..."
    curl -L "https://zenodo.org/records/11237127/files/CTC.zip?download=1" -o CTC.zip

    # Extract the dataset
    echo "Extracting dataset..."
    unzip CTC.zip

    # Clean up
    echo "Cleaning up..."
    rm CTC.zip

    echo "Download complete! Dataset saved in ../datasets/moma"
fi

# Check if YOLO folder exists before converting
if [ -d "yolo" ]; then
    echo "YOLO format dataset already exists, skipping conversion..."
else
    # Run the conversion script
    echo "Converting dataset to YOLO format..."
    python ../../utils/convert_ctc_to_yolo.py
fi

echo "Dataset preparation complete!" 