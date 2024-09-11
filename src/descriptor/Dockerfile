# use simple python image
FROM python:3.8-slim

# set the working directory
WORKDIR /app 

# copy required files to the working directory
COPY src/descriptor/. /app
COPY src/molecules.py src/utils.py src/requirements.txt /app
COPY data/data_cleaned.csv /app
COPY src/descriptor/hyperparameters.yaml /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the scripts are executable
RUN chmod +x trainer.py docker.sh

# Run the bash script when the container launches
#CMD ["/bin/bash", "docker.sh"]

# Go to project parent directory and run docker build -f src/descriptor/Dockerfile -t descr .
