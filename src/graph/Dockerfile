# Use an official Python runtime as a parent image
FROM python:3.10-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/graph/. /app/
COPY data/train_data_cleaned.csv data/test_data_cleaned.csv /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
# COPY src/graph/requirements.txt /app/
RUN pip install --no-cache-dir -r req.txt

# Make sure the scripts are executable
RUN chmod +x trainer.py docker.sh

# Run the bash script when the container launches
CMD ["/bin/bash", "docker.sh"]

# Build command:
# Go to project parent directory and run:
# docker build -f src/graph/Dockerfile -t descr .