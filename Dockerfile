FROM python:3.10-slim

# Set system configurations
WORKDIR /code
ENV PYTHONUNBUFFERED=1

# Install essential system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache layers
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files
COPY ./app /code/app
COPY ./backend /code/backend

# Crucial: Hugging Face Spaces requires containers to run on port 7860
EXPOSE 7860

# Command to boot your FastAPI app on the required port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]