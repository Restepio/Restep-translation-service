# Use the official RunPod Pytorch image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY rp_handler.py .
COPY server.py .
COPY start.py .

# Make start.py executable
RUN chmod +x start.py

# Expose port 8000 for HTTP mode (will be ignored in serverless mode)
EXPOSE 8000

# Run the startup script that detects the environment
CMD ["python", "start.py"]
