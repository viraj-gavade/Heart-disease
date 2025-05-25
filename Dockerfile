FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY application.py ./
COPY Models/ ./Models/
COPY templates/ ./templates/
COPY static/ ./static/
COPY notebooks/ ./notebooks/

# Use a non-root user for better security
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000"]
