"""Optional workspace tokens and signed browser sessions, with no client-side key storage."""

import base64
import hashlib
import hmac
import json
import time
from urllib.parse import urlparse

from fastapi import HTTPException, Request


COOKIE = "studio_session"


def signing_key(settings) -> bytes:
    return hashlib.sha256(json.dumps(settings.access_tokens, sort_keys=True).encode()).digest()


def sign_session(settings, workspace: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({
        "workspace": workspace, "expires": int(time.time()) + 12 * 3600,
    }, separators=(",", ":")).encode()).decode()
    signature = hmac.new(signing_key(settings), payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature


def read_session(settings, cookie: str | None) -> str | None:
    if not cookie:
        return None
    try:
        payload, signature = cookie.split(".", 1)
        expected = hmac.new(signing_key(settings), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        value = json.loads(base64.urlsafe_b64decode(payload))
        if value["expires"] < time.time() or value["workspace"] not in settings.access_tokens:
            return None
        return value["workspace"]
    except (ValueError, KeyError, TypeError):
        return None


def workspace(request: Request) -> str:
    settings = request.app.state.settings
    if not settings.access_tokens:
        return "local"
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        supplied = authorization[7:]
        for name, token in settings.access_tokens.items():
            if hmac.compare_digest(supplied.encode(), token.encode()):
                return name
    name = read_session(settings, request.cookies.get(COOKIE))
    if name:
        return name
    raise HTTPException(401, "ログインが必要です / Sign in to this workspace.")


def require_same_origin(request: Request):
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if request.headers.get("x-studio-request") != "1":
        raise HTTPException(403, "Missing X-Studio-Request header.")
    origin = request.headers.get("origin")
    if origin:
        parsed = urlparse(origin)
        if parsed.netloc != request.headers.get("host") or parsed.scheme not in {"http", "https"}:
            raise HTTPException(403, "Cross-origin writes are not allowed.")
