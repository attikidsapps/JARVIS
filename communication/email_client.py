"""
jarvis/communication/email_client.py

Production SMTP email client for JARVIS.

Handles outbound email (plain text, HTML, and file attachments) via
any SMTP server. Inbound email reading is intentionally out of scope
for this module; that belongs in a future imap_client.py if required.

Security model:
    - Every send operation — with or without attachments — is routed
      through PermissionManager.authorize() with action "email.send"
      (RiskLevel.CRITICAL). JARVIS will never send an email without
      explicit user confirmation.
    - The email password is accepted at construction time and held in
      memory only for the lifetime of the EmailClient instance. It is
      never written to any log, audit record, or database. The
      PermissionManager's metadata sanitizer would redact it anyway,
      but we do not pass it there in the first place.
    - STARTTLS is always negotiated before credentials are sent. Plain
      or SSL-only connections are not supported to prevent accidental
      credential exposure.

Design goals:
    - Stateless per-send: a new SMTP connection is opened and closed
      for each send() call. This avoids stale connection issues without
      requiring a connection pool for the low send-volume of a personal
      assistant.
    - Attachment support is first-class, not bolted on: any number of
      file paths may be attached, with MIME type auto-detected.
    - All public methods return a structured EmailResult rather than a
      bare bool so callers can log the message-id and diagnose failures
      without catching exceptions for control flow.

Dependencies (all in requirements.txt):
    smtplib    -- stdlib, zero cost
    email.*    -- stdlib, zero cost
    mimetypes  -- stdlib, zero cost
    security/permissions.py  -- already written
"""

from __future__ import annotations

import logging
import mimetypes
import smtplib
import ssl
from dataclasses import dataclass, field
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from security.permissions import PermissionDeniedError, PermissionManager

__all__ = [
    "EmailClientError",
    "EmailResult",
    "EmailClient",
]

logger = logging.getLogger("jarvis.communication.email_client")


class EmailClientError(Exception):
    """Raised when an email operation fails for a non-permission reason.

    Permission denials raise PermissionDeniedError from the security
    layer instead — callers should catch both independently so they
    can give the user an accurate explanation.
    """


@dataclass
class EmailResult:
    """Outcome of a send operation.

    Attributes:
        success:     True if the SMTP server accepted the message.
        recipient:   The To: address the message was sent to.
        subject:     The subject line of the message.
        message_id:  The Message-ID header assigned by the SMTP server,
                     if available. Useful for correlating sent mail with
                     delivery receipts and audit logs.
        error:       Human-readable error description if success is False.
    """

    success: bool
    recipient: str
    subject: str
    message_id: Optional[str] = None
    error: Optional[str] = None


class EmailClient:
    """SMTP email client gated behind JARVIS's permission system.

    Usage:
        pm = PermissionManager(confirmation_callback=my_confirm_fn)
        client = EmailClient(
            permission_manager=pm,
            sender_address="me@example.com",
            smtp_password="app-password-here",
        )
        result = client.send(
            recipient="them@example.com",
            subject="Meeting notes",
            body="See attached.",
            attachments=[Path("notes.pdf")],
        )
        if not result.success:
            print(result.error)
    """

    #: Default Gmail SMTP endpoint. Override in constructor for other providers.
    DEFAULT_SMTP_HOST: str = "smtp.gmail.com"
    DEFAULT_SMTP_PORT: int = 587

    def __init__(
        self,
        permission_manager: PermissionManager,
        sender_address: str,
        smtp_password: str,
        smtp_host: str = DEFAULT_SMTP_HOST,
        smtp_port: int = DEFAULT_SMTP_PORT,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        """Initialise the email client.

        Args:
            permission_manager: The shared PermissionManager instance.
                Every send call will call authorize("email.send") on
                this object before opening an SMTP connection.
            sender_address: The From: address used on all outbound mail.
                Must match the account authenticated by smtp_password.
            smtp_password: App password or account password for SMTP
                authentication. For Gmail, generate a dedicated app
                password at myaccount.google.com/apppasswords.
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port. 587 (STARTTLS) is required;
                465 (SSL) and 25 (plain) are not supported.
            connect_timeout_seconds: Timeout for opening the TCP
                connection to the SMTP server.
        """
        self._permission_manager = permission_manager
        self._sender = sender_address
        self._password = smtp_password
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._connect_timeout = connect_timeout_seconds

        logger.info(
            "EmailClient initialised: sender=%r, host=%s:%d",
            self._sender,
            self._smtp_host,
            self._smtp_port,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        recipient: str,
        subject: str,
        body: str,
        html: bool = False,
        attachments: Optional[list[Path]] = None,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
    ) -> EmailResult:
        """Send an email, requesting user confirmation first.

        This is the single public entry point for all outbound email.
        Internally it builds the MIME message, requests permission,
        and only opens the SMTP connection after authorization.

        Args:
            recipient: Primary To: address.
            subject:   Email subject line.
            body:      Email body text. Treated as plain text unless
                       ``html=True``.
            html:      If True, body is sent as text/html. A plain-text
                       fallback containing the same text is included
                       automatically as a multipart/alternative part.
            attachments: Optional list of file Paths to attach. Each
                         file is read at call time; missing files raise
                         EmailClientError before permission is requested.
            cc:        Optional list of CC addresses.
            bcc:       Optional list of BCC addresses. BCC recipients
                       receive the message but are not visible in headers.

        Returns:
            EmailResult with success=True and the server-assigned
            message_id if the send succeeded, or success=False with an
            error description if it failed.

        Raises:
            PermissionDeniedError: If the user declines or the
                permission system times out. Callers must handle this.
            EmailClientError: If attachment files are missing or
                unreadable before the permission prompt is shown.
        """
        attachment_paths = attachments or []

        # Validate all attachment paths BEFORE prompting for permission.
        # There is no point asking the user to confirm sending an email
        # we already know will fail due to a missing file.
        self._validate_attachments(attachment_paths)

        # Request explicit user confirmation — this is a CRITICAL action.
        attachment_summary = (
            f" with {len(attachment_paths)} attachment(s)" if attachment_paths else ""
        )
        self._permission_manager.authorize(
            action="email.send",
            description=(
                f"Send email to {recipient!r} — "
                f"Subject: {subject!r}{attachment_summary}"
            ),
            metadata={
                "recipient": recipient,
                "subject": subject,
                "cc": cc or [],
                "bcc": bcc or [],
                "attachment_count": len(attachment_paths),
                "attachment_names": [p.name for p in attachment_paths],
                "sender": self._sender,
            },
        )
        # If authorize() returns without raising, the user approved.

        message = self._build_message(
            recipient=recipient,
            subject=subject,
            body=body,
            html=html,
            attachments=attachment_paths,
            cc=cc,
            bcc=bcc,
        )

        return self._smtp_send(
            message=message,
            recipient=recipient,
            subject=subject,
            all_recipients=self._collect_all_recipients(recipient, cc, bcc),
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_message(
        self,
        recipient: str,
        subject: str,
        body: str,
        html: bool,
        attachments: list[Path],
        cc: Optional[list[str]],
        bcc: Optional[list[str]],
    ) -> MIMEMultipart:
        """Construct the full MIME message.

        Always uses MIMEMultipart("mixed") as the outer container so
        that attachments and an alternative text/html body can coexist
        without special-casing the no-attachment path.
        """
        outer = MIMEMultipart("mixed")
        outer["From"] = self._sender
        outer["To"] = recipient
        outer["Subject"] = subject

        if cc:
            outer["Cc"] = ", ".join(cc)
        # BCC is intentionally NOT added to headers — that is what makes
        # BCC work. The addresses are passed only to sendmail(), not headers.

        # Body part: wrap in multipart/alternative when HTML is requested
        # so email clients that cannot render HTML fall back to plain text.
        if html:
            alternative = MIMEMultipart("alternative")
            alternative.attach(MIMEText(body, "plain", "utf-8"))
            alternative.attach(MIMEText(body, "html", "utf-8"))
            outer.attach(alternative)
        else:
            outer.attach(MIMEText(body, "plain", "utf-8"))

        # Attachments
        for path in attachments:
            outer.attach(self._build_attachment_part(path))

        return outer

    def _build_attachment_part(self, path: Path) -> MIMEBase:
        """Read a file from disk and return a MIME attachment part.

        Args:
            path: Path to the file to attach.

        Returns:
            A MIMEBase instance ready to attach to the outer message.

        Raises:
            EmailClientError: If the file cannot be read.
        """
        mime_type, encoding = mimetypes.guess_type(str(path))
        if mime_type is None or encoding is not None:
            # Unknown or compressed type — treat as opaque binary.
            mime_type = "application/octet-stream"

        main_type, sub_type = mime_type.split("/", 1)

        try:
            with path.open("rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise EmailClientError(
                f"Cannot read attachment '{path}': {exc}"
            ) from exc

        part = MIMEBase(main_type, sub_type)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=path.name,
        )
        return part

    # ------------------------------------------------------------------
    # SMTP transmission
    # ------------------------------------------------------------------

    def _smtp_send(
        self,
        message: MIMEMultipart,
        recipient: str,
        subject: str,
        all_recipients: list[str],
    ) -> EmailResult:
        """Open an SMTP connection, authenticate, and transmit the message.

        Args:
            message:        The fully-constructed MIME message.
            recipient:      Primary To: address (for the result record).
            subject:        Subject line (for the result record).
            all_recipients: Flattened list of To + CC + BCC addresses.
                            Passed to sendmail() to ensure BCC delivery.

        Returns:
            EmailResult describing the outcome.
        """
        context = ssl.create_default_context()

        try:
            with smtplib.SMTP(
                self._smtp_host,
                self._smtp_port,
                timeout=self._connect_timeout,
            ) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self._sender, self._password)

                refused = server.sendmail(
                    from_addr=self._sender,
                    to_addrs=all_recipients,
                    msg=message.as_string(),
                )

                # sendmail() returns a dict of refused recipients;
                # an empty dict means full success.
                if refused:
                    refused_list = ", ".join(refused.keys())
                    logger.warning(
                        "Email to %r sent but some recipients refused: %s",
                        recipient,
                        refused_list,
                    )

                message_id = message.get("Message-ID")
                logger.info(
                    "Email sent successfully to %r (subject=%r, message_id=%r)",
                    recipient,
                    subject,
                    message_id,
                )
                return EmailResult(
                    success=True,
                    recipient=recipient,
                    subject=subject,
                    message_id=message_id,
                )

        except smtplib.SMTPAuthenticationError as exc:
            error = (
                f"SMTP authentication failed for {self._sender!r}. "
                "Check your app password or account credentials."
            )
            logger.error("Email send failed — auth error: %s", exc)
            return EmailResult(
                success=False, recipient=recipient, subject=subject, error=error
            )

        except smtplib.SMTPRecipientsRefused as exc:
            error = f"All recipients refused by SMTP server: {exc.recipients}"
            logger.error("Email send failed — recipients refused: %s", exc)
            return EmailResult(
                success=False, recipient=recipient, subject=subject, error=error
            )

        except smtplib.SMTPException as exc:
            error = f"SMTP error during send: {exc}"
            logger.error("Email send failed — SMTP error: %s", exc)
            return EmailResult(
                success=False, recipient=recipient, subject=subject, error=error
            )

        except OSError as exc:
            error = (
                f"Network error connecting to {self._smtp_host}:{self._smtp_port}: {exc}"
            )
            logger.error("Email send failed — network error: %s", exc)
            return EmailResult(
                success=False, recipient=recipient, subject=subject, error=error
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_attachments(self, paths: list[Path]) -> None:
        """Raise EmailClientError if any attachment path is missing or unreadable.

        Called before the permission prompt so the user is never asked
        to confirm an operation that is already guaranteed to fail.

        Args:
            paths: List of attachment Paths to validate.

        Raises:
            EmailClientError: On the first missing or unreadable file.
        """
        for path in paths:
            if not path.exists():
                raise EmailClientError(
                    f"Attachment not found: '{path}'. "
                    "Verify the file path and try again."
                )
            if not path.is_file():
                raise EmailClientError(
                    f"Attachment path is not a file: '{path}'."
                )
            try:
                # Probe readability without loading the whole file.
                with path.open("rb") as fh:
                    fh.read(1)
            except OSError as exc:
                raise EmailClientError(
                    f"Cannot read attachment '{path}': {exc}"
                ) from exc

    @staticmethod
    def _collect_all_recipients(
        recipient: str,
        cc: Optional[list[str]],
        bcc: Optional[list[str]],
    ) -> list[str]:
        """Return the deduplicated union of To, CC, and BCC addresses.

        This list is passed to SMTP sendmail() to ensure all recipients
        actually receive the message, while BCC addresses remain absent
        from the visible message headers.
        """
        seen: set[str] = set()
        result: list[str] = []
        for addr in [recipient] + (cc or []) + (bcc or []):
            normalized = addr.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(addr.strip())
        return result