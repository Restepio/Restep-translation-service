# Use the official RunPod Pytorch image
FROM runpod/pytorch:260-py311-cuda1260-cudnn-devel-ubuntu2204

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script to download the model and run it
COPY download_model.py .
RUN python download_model.py

# Copy the handler and main application files
COPY rp_handler.py .
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# Run the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
