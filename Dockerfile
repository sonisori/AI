FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py /app/
COPY src/gpt.py /app/
COPY src/models/model.keras /app/models/

CMD ["python", "app.py"]