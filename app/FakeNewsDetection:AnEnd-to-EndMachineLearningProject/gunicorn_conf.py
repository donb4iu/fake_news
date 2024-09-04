# gunicorn_conf.py

bind = "0.0.0.0:5000"
workers = 4
timeout = 120
loglevel = "debug"
accesslog = "-"
errorlog = "-"
