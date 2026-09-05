FROM python:3.11-slim

WORKDIR /app

COPY website/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--pythonpath", "website", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
