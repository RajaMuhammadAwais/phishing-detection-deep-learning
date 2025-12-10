# Use a stable, official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The model and data preparation scripts are designed to be run sequentially.
# The entrypoint can be a shell script or a single command.
# For simplicity, we'll set the entrypoint to python and let the user specify the script to run.
# Example: docker run <image_name> python data_preparation.py

# To make the container immediately useful, we can set the entrypoint to a script
# that can be used for prediction, but since the goal is to provide the project,
# we'll keep it flexible.

# Expose a port if a web service were to be added (e.g., Flask/FastAPI for prediction)
# EXPOSE 8080

# Define environment variable for TensorFlow to suppress warnings about missing GPU
ENV TF_CPP_MIN_LOG_LEVEL=2

# The command to run the data preparation script
CMD ["python", "data_preparation.py"]
