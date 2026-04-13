FROM python:3.9-slim
WORKDIR /app
# Only copy and install API requirements
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
