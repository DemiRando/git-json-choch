# emailer.py
import smtplib
from email.message import EmailMessage

def send_email(subject, body, sender_email, app_password, recipient_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
            print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Email sending failed: {e}")
