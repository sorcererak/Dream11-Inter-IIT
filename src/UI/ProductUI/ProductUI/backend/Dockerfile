# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set environment variables for better behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PGPASSWORD=Ak@123

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for PostgreSQL client and clean up afterward
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Copy and set up the entrypoint script
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the application's default port
EXPOSE 80

# Run the entrypoint script when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]
