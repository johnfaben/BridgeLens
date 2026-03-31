from authlib.integrations.requests_client import OAuth2Session
from flask import current_app, url_for, request, redirect, session


class OAuthSignIn(object):
    providers = None

    def __init__(self, provider_name):
        self.provider_name = provider_name
        credentials = current_app.config['OAUTH_CREDENTIALS'][provider_name]
        self.consumer_id = credentials['id']
        self.consumer_secret = credentials['secret']

    def authorize(self):
        pass

    def callback(self):
        pass

    def get_callback_url(self):
        return url_for('oauth_callback', provider=self.provider_name, _external=True)

    @classmethod
    def get_provider(cls, provider_name):
        if cls.providers is None:
            cls.providers = {}
            for provider_class in cls.__subclasses__():
                provider = provider_class()
                cls.providers[provider.provider_name] = provider
        return cls.providers[provider_name]


class GoogleSignIn(OAuthSignIn):
    AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
    TOKEN_URL = 'https://oauth2.googleapis.com/token'
    USERINFO_URL = 'https://openidconnect.googleapis.com/v1/userinfo'

    def __init__(self):
        super(GoogleSignIn, self).__init__('google')

    def authorize(self):
        oauth = OAuth2Session(
            client_id=self.consumer_id,
            client_secret=self.consumer_secret,
            redirect_uri=self.get_callback_url(),
            scope='openid email profile',
        )
        uri, state = oauth.create_authorization_url(self.AUTHORIZE_URL)
        session['oauth_state'] = state
        return redirect(uri)

    def callback(self):
        if 'code' not in request.args:
            return None, None, None

        oauth = OAuth2Session(
            client_id=self.consumer_id,
            client_secret=self.consumer_secret,
            redirect_uri=self.get_callback_url(),
            state=session.pop('oauth_state', None),
        )
        token = oauth.fetch_token(
            self.TOKEN_URL,
            authorization_response=request.url,
        )
        resp = oauth.get(self.USERINFO_URL)
        me = resp.json()

        return (
            me.get('email', '').split('@')[0],
            me.get('email'),
            me.get('name'),
        )
