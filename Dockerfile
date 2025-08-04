# Use the official RunPod Pytorch image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app


# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY rp_handler.py .


# Run the Uvicorn server with multiple workers
CMD ["python", "rp_handler.py"]
