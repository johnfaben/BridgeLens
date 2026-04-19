from functools import wraps
from threading import Thread

from flask import abort
from flask_login import current_user


def run_async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        from app import app
        if not current_user.is_authenticated:
            abort(404)
        if current_user.email not in app.config.get('ADMINS', []):
            abort(404)
        return f(*args, **kwargs)
    return wrapper
