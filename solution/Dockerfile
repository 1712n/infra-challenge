# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files into the Docker image
COPY requirements.txt /app
COPY main.py /app
COPY models.py /app

# Install dependencies
RUN pip install -r requirements.txt
# Set environment variables
ENV FASTAPI_APP=main.py

# Expose port
EXPOSE 8000

# RUN python3 models.py
# Command to run the Flask application
ENTRYPOINT [ "bash", "build_start.sh" ]
