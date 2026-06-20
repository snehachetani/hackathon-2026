"""
Fetch invoice attachments from Gmail via IMAP.
Downloads PDF/PNG/DOCX attachments from unread emails into a local folder.
"""

import imaplib
import email
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}


def decode_mime_words(s: str) -> str:
    parts = decode_header(s)
    return "".join(
        part.decode(enc or "utf-8") if isinstance(part, bytes) else part
        for part, enc in parts
    )


def fetch_invoice_attachments(
    email_address: str,
    app_password: str,
    output_dir: Path,
    folder: str = "INBOX",
    last_minutes: int = 10,
) -> list[Path]:
    """
    Connects to Gmail, finds unread emails that arrived in the last
    `last_minutes` minutes, downloads invoice attachments to output_dir.
    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)
    since_str = cutoff.strftime("%d-%b-%Y")  # IMAP SINCE is date-only

    print(f"  Connecting to Gmail ({email_address})...")
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as mail:
        mail.login(email_address, app_password)
        mail.select(folder)

        # Server-side filter: unread + today's date (fast, no body download)
        status, msg_ids = mail.search(None, "UNSEEN", f'SINCE "{since_str}"')
        if status != "OK" or not msg_ids[0]:
            print("  No new unread emails today.")
            return []

        ids = msg_ids[0].split()
        print(f"  {len(ids)} unread email(s) today — checking which arrived in last {last_minutes} min...")

        for msg_id in ids:
            # Headers-only fetch to check exact timestamp before downloading body
            status, hdr_data = mail.fetch(msg_id, "(BODY[HEADER.FIELDS (DATE FROM SUBJECT)])")
            if status != "OK":
                continue

            hdr_msg = email.message_from_bytes(hdr_data[0][1])
            date_str = hdr_msg.get("Date", "")
            subject = decode_mime_words(hdr_msg.get("Subject", "(no subject)"))

            try:
                msg_time = parsedate_to_datetime(date_str)
                if msg_time.tzinfo is None:
                    msg_time = msg_time.replace(tzinfo=timezone.utc)
            except Exception:
                msg_time = datetime.now(timezone.utc)

            if msg_time < cutoff:
                print(f"  Skipped (too old): {subject[:55]}")
                continue

            # In the time window — now fetch the full email
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject = decode_mime_words(msg.get("Subject", "(no subject)"))
            sender = msg.get("From", "unknown")
            print(f"  Found: {subject[:55]}  [{sender[:30]}]")

            attachments_found = 0
            for part in msg.walk():
                content_disposition = part.get("Content-Disposition", "")
                if "attachment" not in content_disposition:
                    continue

                filename = part.get_filename()
                if not filename:
                    continue
                filename = decode_mime_words(filename)
                ext = Path(filename).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue

                # Avoid name collisions
                save_path = output_dir / filename
                counter = 1
                while save_path.exists():
                    save_path = output_dir / f"{Path(filename).stem}_{counter}{ext}"
                    counter += 1

                payload = part.get_payload(decode=True)
                if payload:
                    save_path.write_bytes(payload)
                    downloaded.append(save_path)
                    attachments_found += 1
                    print(f"  Downloaded: {save_path.name}  (from: {sender} | {subject})")

            if attachments_found > 0:
                # Mark as read so we don't reprocess
                mail.store(msg_id, "+FLAGS", "\\Seen")

    print(f"  Total attachments downloaded: {len(downloaded)}")
    return downloaded
