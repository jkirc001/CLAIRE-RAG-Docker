#!/bin/bash
set -e

DEST_DIR="./vectorstore"
RELEASE_URL="https://github.com/jkirc001/CLAIRE-RAG-Docker/releases/download/v0.1.0-data/vectorstore.tar.gz"

if [ -f "$DEST_DIR/chroma.sqlite3" ]; then
    echo "Vectorstore already exists. Skipping download."
    exit 0
fi

echo "Downloading vectorstore..."
curl -L --progress-bar -o /tmp/vectorstore.tar.gz "$RELEASE_URL"

echo "Extracting..."
tar xzf /tmp/vectorstore.tar.gz
rm /tmp/vectorstore.tar.gz

echo "Download complete. Vectorstore at $DEST_DIR/"
