# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app code
COPY ./app /code/app

# Run the API using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]