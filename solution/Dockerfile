# Base image
FROM python:3.9

# Copy project files into the Docker image
COPY requirements.txt /app
COPY main.py /app
COPY models.py /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV FASTAPI_APP=main.py

# Expose port
EXPOSE 8000

RUN python3 models.py
# Command to run the Flask application
CMD ["uvicorn", "main:app", "--reload"]
