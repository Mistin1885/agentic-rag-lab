import logging

# Suppress httpx HTTP request logs to avoid leaking internal endpoint URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
