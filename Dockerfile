FROM python:3.10-slim

# System packages needed for Chromium
RUN apt-get update && apt-get install -y \
    wget curl gnupg unzip fonts-liberation libatk-bridge2.0-0 libnss3 libxss1 libasound2 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxshmfence1 libgtk-3-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Workdir + code
WORKDIR /app
COPY . .

# Install Python deps including latest Playwright
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 🧠 DON'T set PLAYWRIGHT_BROWSERS_PATH — use default path!
RUN python -m playwright install chromium

# Render requires this
ENV PORT=10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
