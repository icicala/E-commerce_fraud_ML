FROM python:3.11

# working directory
WORKDIR /app

# cp the necessary files
COPY fastAPI_model.py .
COPY binary_encoder.joblib .
COPY RFC_model.joblib .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "fastAPI_model:fastAPI_model", "--host", "0.0.0.0", "--port", "8000"]