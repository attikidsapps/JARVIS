"""Email client for sending and receiving emails."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional


class EmailClient:
    """Handles email operations."""

    def __init__(self, email: str, password: str, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        """
        Initialize email client.
        
        Args:
            email: Email address
            password: Email password or app password
            smtp_server: SMTP server address
            smtp_port: SMTP server port
        """
        self.email = email
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email(self, recipient: str, subject: str, body: str, html: bool = False) -> bool:
        """
        Send an email.
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body
            html: Whether body is HTML
            
        Returns:
            True if successful, False otherwise
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email
            msg["To"] = recipient
            msg["Subject"] = subject

            mime_type = "html" if html else "plain"
            msg.attach(MIMEText(body, mime_type))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def send_email_with_attachment(self, recipient: str, subject: str, body: str, attachment_path: str) -> bool:
        """Send email with attachment."""
        # TODO: Implement attachment support
        pass
