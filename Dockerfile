# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install build-essential, x11-apps, and clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    x11-apps \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch
RUN pip install deap

# Install other dependencies
RUN pip install numpy pygame pymunk debugpy
# Install PyTorch
RUN pip install torch torchvision

# Install other dependencies
RUN pip install numpy pygame debugpy

# Set the working directory
WORKDIR /pyDEAP

# Copy your project files into the container
COPY . /pyDEAP

# Set an environment variable for the script name
ENV SCRIPT_NAME=main.py

# Ensure the script is executable
RUN chmod +x /pyDEAP/$SCRIPT_NAME

# Set the DISPLAY environment variable
ENV DISPLAY=host.docker.internal:0.0

# Expose the debug port
EXPOSE 5680

# Run your main script
# Run your main script and keep the container running
CMD ["sh", "-c", "python /pyDEAP/$SCRIPT_NAME & tail -f /dev/null"]