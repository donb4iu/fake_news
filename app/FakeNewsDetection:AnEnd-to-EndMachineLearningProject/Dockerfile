# Use the official Python 3.8 slim image as the base image
FROM python:3.12.3-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /api-flask

# Copy the necessary files and directories into the container
COPY . /api-flask

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

# Expose port 5000 for the Flask application
EXPOSE 5000

# Define the command to run the Flask application using Gunicorn
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app", "-b", "0.0.0.0:5000", "-w", "4"]
