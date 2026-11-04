# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for TensorFlow and MySQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces requires the app to run on port 7860
EXPOSE 7860

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]