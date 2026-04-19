import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

database_url = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'app.db'))
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

SQLALCHEMY_DATABASE_URI = database_url
SQLALCHEMY_TRACK_MODIFICATIONS = False

WTF_CSRF_ENABLED = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

OAUTH_CREDENTIALS = {
    'google': {
        'id': os.environ.get('GOOGLE_CLIENT_ID', ''),
        'secret': os.environ.get('GOOGLE_CLIENT_SECRET', ''),
    },
}

RESEND_API_KEY = os.environ.get('RESEND_API_KEY')

ADMINS = [e.strip() for e in os.environ.get('ADMIN_EMAILS', os.environ.get('ADMIN_EMAIL', '')).split(',') if e.strip()]

MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB upload limit
