# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    # Set the data directory path used within the container
    APP_DATA_DIR=/app/data

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# build-essential needed for some pip installs, ffmpeg for video/audio, imagemagick for moviepy text
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    imagemagick \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import moviepy.editor; print('Successfully imported moviepy.editor during build!')"

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of the application code into the container
COPY . .

# Ensure the data directory exists (though volume mount will likely create it)
# RUN mkdir -p $APP_DATA_DIR

# Expose the port the app runs on
EXPOSE 5000

# Specify the entrypoint script
ENTRYPOINT ["entrypoint.sh"]

# Default command (gets executed by entrypoint script)
# CMD ["flask", "run"] # We use exec in entrypoint.sh instead