import json
import uuid

from flask import session, request
from flask_login import current_user

from app import app, db
from app.models import Event


SESSION_KEY = '_aid'


def _session_id():
    sid = session.get(SESSION_KEY)
    if not sid:
        sid = uuid.uuid4().hex
        session[SESSION_KEY] = sid
        session.permanent = True
    return sid


def log_event(event_type, upload_id=None, data=None, user_id=None, session_id=None):
    """Record an analytics event. Adds to the current db.session but does not commit.

    If called from a normal request, the caller's request lifecycle (or a
    following db.session.commit) will flush it. From the SSE generator pass
    an explicit user_id / session_id captured before the generator starts,
    and commit within the generator's own scoped session context.
    """
    try:
        if session_id is None:
            session_id = _session_id()
        if user_id is None and current_user.is_authenticated:
            user_id = current_user.id
        ev = Event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            upload_id=upload_id,
            data=json.dumps(data) if data else None,
        )
        db.session.add(ev)
    except Exception as e:
        app.logger.warning('log_event failed: %s', e)


def log_event_commit(event_type, upload_id=None, data=None, user_id=None, session_id=None):
    """Same as log_event but commits immediately. Use inside the SSE generator."""
    try:
        ev = Event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            upload_id=upload_id,
            data=json.dumps(data) if data else None,
        )
        db.session.add(ev)
        db.session.commit()
    except Exception as e:
        app.logger.warning('log_event_commit failed: %s', e)
        db.session.rollback()


def avg_inference_seconds(default=12.0, n=50):
    """Average end-to-end inference time from recent completed events."""
    try:
        rows = (Event.query
                .filter_by(event_type='inference_completed')
                .order_by(Event.id.desc())
                .limit(n).all())
        times = []
        for r in rows:
            d = r.get_data()
            t = (d.get('time_detect') or 0) + (d.get('time_classify') or 0)
            if t > 0:
                times.append(t)
        if times:
            return sum(times) / len(times)
    except Exception as e:
        app.logger.warning('avg_inference_seconds failed: %s', e)
    return default
