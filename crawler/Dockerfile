FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

WORKDIR /app

COPY crawler/requirements.txt ./crawler/requirements.txt
RUN pip install --no-cache-dir -r crawler/requirements.txt
RUN playwright install-deps chromium && playwright install chromium

COPY . .

ENV PYTHONPATH=/app
CMD ["python", "crawler/main.py"]
