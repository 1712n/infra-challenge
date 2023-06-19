# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the desired port (change it if needed)
EXPOSE 8000

# Set environment variables if necessary
ENV ENV_VAR_NAME=value

# Start the application when the container launches
CMD ["python", "app.py"]
