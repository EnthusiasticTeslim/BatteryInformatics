# Use an official Python runtime as a parent image
FROM python:3.10-bookworm

# set the working directory
WORKDIR /app 

# copy required files to the working directory
COPY docker/descriptor/. /app/
COPY src/descriptor/. /app
COPY data/train_data_cleaned.csv data/test_data_cleaned.csv /app/
COPY regenerate/hp_descriptor.yaml /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the scripts are executable
RUN chmod +x trainer.py docker.sh

# Run the bash script when the container launches
CMD ["/bin/bash", "docker.sh"]

# Build command: Go to project parent directory and run: 
#       docker build -f docker/descriptor/Dockerfile -t descriptor .
# Run command: 
#       docker run -it --rm --name descriptor descriptor