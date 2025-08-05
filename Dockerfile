# Use the official RunPod Pytorch image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI server file
COPY rp_handler.py .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "rp_handler:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
