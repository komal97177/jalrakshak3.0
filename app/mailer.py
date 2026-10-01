import os, smtplib, logging
from email.message import EmailMessage
log = logging.getLogger("jalrakshak.mail")


def configured(): return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER") and os.getenv("SMTP_PASS"))


def send_many(messages):
    """messages: list of (to, subject, body). Returns number really sent (0 in dry-run mode)."""
    if not configured():
        for to, sub, _ in messages: log.warning("DRY-RUN email to %s***: %s", to[:2], sub)
        return 0
    sent = 0
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587")), timeout=20) as s:
        s.starttls(); s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        for to, sub, body in messages:
            m = EmailMessage(); m["From"] = os.getenv("MAIL_FROM", os.environ["SMTP_USER"]); m["To"] = to; m["Subject"] = sub; m.set_content(body)
            try: s.send_message(m); sent += 1
            except Exception as e: log.error("send failed: %s", e)
    return sent
