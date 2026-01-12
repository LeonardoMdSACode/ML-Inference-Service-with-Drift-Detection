# Helper functions for sending notifications.

import smtplib
from email.message import EmailMessage
import requests

def send_email_alert(message: str):
    # Configure your SMTP settings here
    try:
        email = EmailMessage()
        email.set_content(message)
        email["Subject"] = "ML Governance Alert"
        email["From"] = "ml.alerts@example.com"
        email["To"] = "ops-team@example.com"

        with smtplib.SMTP("localhost") as smtp:
            smtp.send_message(email)
    except Exception as e:
        print(f"Failed to send email alert: {e}")


def send_slack_alert(message: str):
    # Slack webhook URL
    webhook_url = "https://hooks.slack.com/services/XXXX/YYYY/ZZZZ"
    try:
        requests.post(webhook_url, json={"text": message})
    except Exception as e:
        print(f"Failed to send Slack alert: {e}")
