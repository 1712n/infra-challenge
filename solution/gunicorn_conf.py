import multiprocessing

# Server Socket
bind = '0.0.0.0:8000'

# Worker Processes
workers = 2
# multiprocessing.cpu_count() * 2 + 1
worker_class = 'gthread'

# The maximum number of simultaneous clients.
worker_connections = 5000

# Server Mechanics
preload_app = True
timeout = 600  # Timeout in seconds
keepalive = 600  # Keep-alive value in seconds

# Logging
accesslog = '-'  # '-' means log to stdout
errorlog = '-'  # '-' means log to stderr
loglevel = 'debug'
