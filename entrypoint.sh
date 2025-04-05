#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Optional: Add wait loop here if you have a separate DB container
# echo "Waiting for database..."
# Add logic to wait for DB if needed

echo "Applying database migrations..."
# Navigate to the app directory if needed (WORKDIR should handle this)
# cd /app
flask db upgrade

echo "Starting Flask application..."
# Use exec to replace the shell process with the Flask process.
# This ensures signals (like Ctrl+C) are passed correctly to Flask.
exec flask run --host=0.0.0.0 --port=5000