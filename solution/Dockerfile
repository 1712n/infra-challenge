# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files into the Docker image
COPY requirements.txt /app
COPY main.py /app
COPY models.py /app
RUN free -g

# Install dependencies
RUN pip install -r requirements.txt
RUN python3 models.py
# Set environment variables
ENV FASTAPI_APP=main.py

# Expose port
EXPOSE 8000

# RUN python3 models.py
# Command to run the Flask application
CMD ["uvicorn", "main:app", "--workers=2", "--worker-class=uvicorn.workers.UvicornWorker"]
