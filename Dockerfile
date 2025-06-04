# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

RUN chmod +x /app/entrypoint.sh

# Define the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]