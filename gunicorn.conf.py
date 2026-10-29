"""Automatically read by the existing `gunicorn app:app` Render command."""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1  # Never load a separate 600M model in multiple worker processes.
threads = 4  # Keep health checks responsive while one translation runs.
timeout = 300
accesslog = '-'
errorlog = '-'
