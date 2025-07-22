FROM python:3.10-slim

# Install system dependencies needed for Chromium
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl \
    libnss3 libatk-bridge2.0-0 libxss1 libasound2 libgtk-3-0 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxshmfence1 libglu1-mesa \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable so Playwright installs browsers into app dir
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright

# Set working directory
WORKDIR /app

# Copy files and install dependencies
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Chromium browser into project path
RUN python -m playwright install chromium

# Expose port for Render
ENV PORT=10000

# Start your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
