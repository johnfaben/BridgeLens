import json
from datetime import datetime, timezone
from app import db
from flask_login import UserMixin


class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    display_name = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = db.Column(db.DateTime)
    uploads = db.relationship('Upload', backref='user', lazy='dynamic',
                              order_by='Upload.created_at.desc()')

    def get_id(self):
        return str(self.id)

    @staticmethod
    def make_unique_username(username):
        if User.query.filter_by(username=username).first() is None:
            return username
        version = 2
        while True:
            new_username = username + str(version)
            if User.query.filter_by(username=new_username).first() is None:
                break
            version += 1
        return new_username

    def __repr__(self):
        return '<User %r>' % self.username


class Upload(db.Model):
    __tablename__ = 'upload'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    original_filename = db.Column(db.String(256))
    stored_filename = db.Column(db.String(256))
    result_filename = db.Column(db.String(256))
    pbn = db.Column(db.String(512))
    bbo_url = db.Column(db.String(1024))
    total_cards = db.Column(db.Integer)
    training_consent = db.Column(db.Boolean, default=False)
    # JSON blob: list of {bbox: [x1,y1,x2,y2], class_name: str, confidence: float}
    detections_json = db.Column(db.Text)
    # JSON blob: same format, after user corrections (null if uncorrected)
    corrections_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def get_detections(self):
        if not self.detections_json:
            return []
        return json.loads(self.detections_json)

    def set_detections(self, detections):
        self.detections_json = json.dumps(detections)

    def get_corrections(self):
        if not self.corrections_json:
            return None
        return json.loads(self.corrections_json)

    def set_corrections(self, corrections):
        self.corrections_json = json.dumps(corrections)

    def __repr__(self):
        return '<Upload %r>' % self.original_filename


class Event(db.Model):
    __tablename__ = 'event'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    event_type = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    session_id = db.Column(db.String(64), index=True, nullable=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('upload.id'), nullable=True, index=True)
    data = db.Column(db.Text, nullable=True)

    def get_data(self):
        if not self.data:
            return {}
        return json.loads(self.data)

    def __repr__(self):
        return '<Event %s>' % self.event_type
