# Use a base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy project files into the Docker image
COPY requirements.txt /app
COPY app.py /app
COPY inference.py /app
COPY models.py /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port on which the application will run
EXPOSE 8000

# Set the command to run the application when the container starts
# CMD ["python", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


