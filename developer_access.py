"""Local password storage for entering the imaging GUI's Developer mode."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets


class DeveloperPasswordStore:
    ITERATIONS = 600_000

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def is_configured(self) -> bool:
        return self.path.exists()

    def set_password(self, password: str) -> None:
        if not password:
            raise ValueError("The developer password cannot be empty.")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, self.ITERATIONS)
        record = {"version": 1, "salt": salt.hex(), "digest": digest.hex()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never replace a password during first-time setup, even in another app instance.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")

    def verify(self, password: str) -> bool:
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if record["version"] != 1:
                raise ValueError("Unsupported password record version.")
            salt = bytes.fromhex(record["salt"])
            expected = bytes.fromhex(record["digest"])
            if len(salt) != 16 or len(expected) != 32:
                raise ValueError("Invalid password record.")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The saved developer password is invalid; Developer mode remains locked.") from exc
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, self.ITERATIONS)
        return hmac.compare_digest(actual, expected)
