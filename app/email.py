import requests
from flask import render_template
from app import app
from .decorators import run_async

RESEND_API_URL = 'https://api.resend.com/emails'
FROM_ADDRESS = 'BridgeLens <noreply@johnfaben.com>'


@run_async
def _send_async(app_ctx, payload):
    with app_ctx:
        api_key = app.config.get('RESEND_API_KEY')
        if not api_key:
            return
        resp = requests.post(
            RESEND_API_URL,
            headers={'Authorization': 'Bearer ' + api_key},
            json=payload,
        )
        if resp.status_code != 200:
            app.logger.error('Resend API error %s: %s', resp.status_code, resp.text)


def send_email(subject, recipients, text_body, html_body):
    payload = {
        'from': FROM_ADDRESS,
        'to': recipients,
        'subject': subject,
        'text': text_body,
        'html': html_body,
    }
    _send_async(app.app_context(), payload)


def send_magic_link(email, token):
    from flask import url_for
    link = url_for('magic_link_verify', token=token, _external=True)
    if not app.config.get('RESEND_API_KEY'):
        app.logger.info('Magic link for %s: %s', email, link)
        print('\n*** Magic link for %s: %s ***\n' % (email, link))
        return
    send_email(
        '[BridgeLens] Your login link',
        [email],
        render_template('magic_link_email.txt', link=link),
        render_template('magic_link_email.html', link=link),
    )
