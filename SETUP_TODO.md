# BridgeLens — Setup TODO

Things to do outside of code before deploying.

---

## 1. Google OAuth

In the [Google Cloud Console](https://console.cloud.google.com/):

1. **APIs & Services > Credentials** — either use the existing OAuth client
   (from hipegame) or create a new one
2. Add to **Authorized redirect URIs**:
   - `https://bridgelens.johnfaben.com/callback/google` (production)
   - `http://127.0.0.1:5000/callback/google` (local dev, optional)
3. Note down the **Client ID** and **Client Secret** for the `.env` file

---

## 2. Resend (magic link emails)

Your `johnfaben.com` domain should already be verified in Resend from hipegame.
If not:

1. Sign in at [resend.com](https://resend.com)
2. **Domains** > Add & verify `johnfaben.com` (DNS records)
3. **API Keys** > Create a key (or reuse the hipegame one)

---

## 3. Hetzner server — database

SSH into the server and create the PostgreSQL database:

```bash
sudo -u postgres psql
```

```sql
CREATE USER bridgelens WITH PASSWORD 'PICK_A_STRONG_PASSWORD';
CREATE DATABASE bridgelens OWNER bridgelens;
\q
```

---

## 4. Server `.env` file

Create `/var/www/bridgelens/.env`:

```env
SECRET_KEY=<generate with: python3 -c "import secrets; print(secrets.token_hex(32))">
DATABASE_URL=postgresql://bridgelens:PICK_A_STRONG_PASSWORD@localhost/bridgelens
GOOGLE_CLIENT_ID=<from step 1>
GOOGLE_CLIENT_SECRET=<from step 1>
RESEND_API_KEY=<from step 2>
```

```bash
chmod 600 /var/www/bridgelens/.env
```

---

## 5. Copy model files to server

From local machine:

```bash
scp "/c/Users/jdfab/Google Drive/bridgelens/best_corner_detector.pt" \
    "/c/Users/jdfab/Google Drive/bridgelens/best_corner_classifier_cnn.pt" \
    deploy@johnfaben-cloud1:/var/www/bridgelens/
```

---

## 6. Run database migration on server

After deploying the code:

```bash
cd /var/www/bridgelens
source venv/bin/activate
flask db upgrade
```

---

## Notes

- **Local dev**: magic links print to the terminal (no Resend key needed).
  Google OAuth won't work locally unless you add the localhost redirect URI
  in step 1.
- **DEPLOY.md** has the full Nginx/systemd/SSL setup instructions.
