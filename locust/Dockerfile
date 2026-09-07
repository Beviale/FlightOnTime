FROM python:3.12-slim-bookworm

WORKDIR /mnt/locust

RUN pip install --no-cache-dir locust

EXPOSE 8089


CMD ["sh", "-c", "locust --host=\"$TARGET_HOST\" --web-host=0.0.0.0 --html=/reports/report.html --csv=/reports/run --csv-full-history"]
