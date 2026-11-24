FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STUDIO_DATA_DIR=/app/data
WORKDIR /app
RUN groupadd --gid 10001 studio && useradd --uid 10001 --gid studio --create-home studio
COPY pyproject.toml requirements.lock README.md LICENSE ./
COPY studio ./studio
RUN pip install --no-cache-dir -c requirements.lock . && mkdir -p /app/data && chown -R studio:studio /app
USER studio
EXPOSE 8000
CMD ["jev-studio", "serve", "--host", "0.0.0.0", "--port", "8000"]
