#!/bin/bash
# Script to download PCD files from AWS S3 bucket

# Delete old data
rm -r original_pc
mkdir original_pc

# Download data
aws s3 cp s3://3d-data-marble/texas-tech/3d-models/ ./original_pc/ --recursive
