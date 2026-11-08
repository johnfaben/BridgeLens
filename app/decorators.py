from functools import wraps
from threading import Thread

from flask import abort
from flask_login import current_user


def run_async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper


def local_tools_enabled():
    """True under the local debug server, or when ENABLE_CORRECTIONS=1."""
    from app import app
    return app.debug or app.config.get('CORRECTIONS_ENABLED', False)


def local_only(f):
    """404 unless local tools are enabled.

    For the correction workflow (/correct...), which writes training labels and
    lists every upload; users fix their hands on the live site via /edit.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not local_tools_enabled():
            abort(404)
        return f(*args, **kwargs)
    return wrapper


def is_admin():
    from app import app
    return current_user.is_authenticated and current_user.email in app.config.get('ADMINS', [])


def can_modify_upload(upload):
    """Whether the current visitor may edit this upload's hands.

    Allowed: local tools, admins, the logged-in uploader, or (for anonymous
    uploads) the browser session that submitted it.
    """
    from app.models import Event
    from app.analytics import SESSION_KEY
    from flask import session
    if upload is None:
        return False
    if local_tools_enabled() or is_admin():
        return True
    if upload.user_id is not None:
        return current_user.is_authenticated and current_user.id == upload.user_id
    sid = session.get(SESSION_KEY)
    if not sid:
        return False
    return Event.query.filter_by(event_type='upload_submitted', upload_id=upload.id,
                                 session_id=sid).first() is not None


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
