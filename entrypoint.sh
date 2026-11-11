#!/bin/sh
# Serve with gunicorn, never the Flask development server (its debugger is a
# remote Python console). One worker: reports are kept in process memory, so a
# second worker would not find them; threads handle concurrent requests.
exec gunicorn \
  --bind "${FLASK_HOST:-0.0.0.0}:${FLASK_PORT:-5000}" \
  --workers 1 \
  --threads "${GUNICORN_THREADS:-4}" \
  --timeout "${GUNICORN_TIMEOUT:-180}" \
  --access-logfile - \
  app:app
