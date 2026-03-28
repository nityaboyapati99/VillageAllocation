FROM python:3.11-slim

WORKDIR /app

# Install deps fast — no cache, prefer binary wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

# Pre-generate village data at build time
RUN python -c "from environment.village_generator import generate_and_save_all; generate_and_save_all()"

EXPOSE 7860

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
