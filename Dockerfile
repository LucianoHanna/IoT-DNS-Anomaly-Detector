FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY src/*.py .

RUN mkdir -p /app/data /app/models /app/results

VOLUME ["/app/data", "/app/models", "/app/results"]

ENTRYPOINT ["/app/entrypoint.sh"]