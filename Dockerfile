FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl gnupg unzip fonts-liberation libnss3 libatk-bridge2.0-0 libgtk-3-0 \
    libdrm2 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
