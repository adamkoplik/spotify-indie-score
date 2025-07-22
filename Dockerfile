FROM python:3.10-slim

# Install system dependencies required by Playwright and Chromium
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl \
    libnss3 libatk-bridge2.0-0 libxss1 libasound2 libgtk-3-0 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxshmfence1 libglu1-mesa \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Playwright and Chromium browser
RUN python -m playwright install --with-deps chromium

# Expose the port for Render
ENV PORT=10000

# Start your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
