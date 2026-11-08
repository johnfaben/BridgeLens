---
name: restart-flask
description: Kill and restart the local Flask development server on port 5000
---

Restart the local Flask dev server reliably:

1. Kill any process listening on port 5000:
```bash
PID=$(netstat -ano | grep ':5000.*LISTENING' | awk '{print $NF}' | head -1)
[ -n "$PID" ] && taskkill //F //PID $PID
```

2. Wait until port 5000 is actually free (poll, don't just sleep):
```bash
for i in $(seq 1 10); do
  netstat -ano | grep ':5000.*LISTENING' > /dev/null || break
  sleep 1
done
```

3. Start the Flask app in the background:
```bash
cd "/c/Users/jdfab/Google Drive/bridgelens" && python run.py
```
(Use the Bash tool with `run_in_background: true`)

4. Wait until the app is actually responding:
```bash
for i in $(seq 1 20); do
  curl -s http://127.0.0.1:5000/ > /dev/null 2>&1 && break
  sleep 1
done
```

5. Confirm to the user whether the restart succeeded or failed.
