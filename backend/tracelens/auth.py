"""Authentication primitives shared by HTTP and MCP integrations."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .storage import get_user_by_id

try:
    import redis
except ImportError:  # pragma: no cover - optional in local-only mode
    redis = None


bearer = HTTPBearer(auto_error=False)
_local_rate_limits: dict[str, tuple[int, float]] = {}
_local_revoked_tokens: dict[str, float] = {}


def _secret() -> bytes:
    return os.getenv("JWT_SECRET", "development-only-change-me").encode("utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)
    return "pbkdf2_sha256$240000$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_value, digest_value = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_value.encode())
        expected = base64.urlsafe_b64decode(digest_value.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(rounds))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _encode_segment(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _decode_segment(value: str) -> dict[str, Any]:
    padded = value + "=" * (-len(value) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded.encode()))
    if not isinstance(decoded, dict):
        raise ValueError("invalid token payload")
    return decoded


def create_access_token(user: dict[str, Any]) -> str:
    now = int(time.time())
    expires_at = now + int(os.getenv("JWT_EXPIRE_HOURS", "24")) * 3600
    token_id = secrets.token_urlsafe(24)
    header = _encode_segment({"alg": "HS256", "typ": "JWT"})
    payload = _encode_segment({
        "sub": str(user["id"]), "username": user["username"], "role": user.get("role", "USER"),
        "jti": token_id, "iat": now, "exp": expires_at,
    })
    unsigned = f"{header}.{payload}".encode()
    signature = base64.urlsafe_b64encode(hmac.new(_secret(), unsigned, hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"{header}.{payload}.{signature}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header, payload, signature = token.split(".", 2)
        unsigned = f"{header}.{payload}".encode()
        expected = base64.urlsafe_b64encode(hmac.new(_secret(), unsigned, hashlib.sha256).digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid signature")
        header_claims = _decode_segment(header)
        if header_claims.get("alg") != "HS256" or header_claims.get("typ") != "JWT":
            raise ValueError("unsupported token header")
        claims = _decode_segment(payload)
        if not claims.get("jti") or not claims.get("sub"):
            raise ValueError("missing token claims")
        if int(claims.get("exp", 0)) <= int(time.time()):
            raise ValueError("expired token")
        token_id = str(claims["jti"])
        expires_at = float(claims.get("exp", 0))
        if _local_revoked_tokens.get(token_id, 0) > time.time():
            raise ValueError("revoked token")
        _local_revoked_tokens.pop(token_id, None)
        _check_redis_revocation(token_id)
        return claims
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError, KeyError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token")


def _redis_client() -> Any | None:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis is None or not redis_url:
        return None
    try:
        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        )
        client.ping()
        return client
    except Exception:
        return None


def _check_redis_revocation(token_id: str) -> None:
    client = _redis_client()
    if client is not None and client.get(f"jwt:revoked:{token_id}"):
        raise ValueError("revoked token")


def revoke_access_token(token: str) -> None:
    claims = decode_access_token(token)
    token_id = str(claims["jti"])
    expires_at = float(claims.get("exp", time.time()))
    ttl = max(1, int(expires_at - time.time()))
    _local_revoked_tokens[token_id] = expires_at
    client = _redis_client()
    if client is not None:
        try:
            client.setex(f"jwt:revoked:{token_id}", ttl, "1")
        except Exception:
            pass


def require_authenticated_user(user: dict[str, Any] | None) -> dict[str, Any]:
    """Require a user only when the instance is running in protected mode."""
    if os.getenv("AUTH_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"} and user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    return user or {}


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    claims = decode_access_token(credentials.credentials)
    try:
        user_id = int(claims["sub"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token") from error
    user = get_user_by_id(user_id)
    if user is None or not user.get("is_active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user is inactive or missing")
    return user


def optional_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any] | None:
    if credentials is None:
        return None
    return current_user(credentials)


def enforce_login_rate_limit(request: Request, username: str) -> None:
    key = f"tracelens:login:{request.client.host if request.client else 'unknown'}:{username.lower()}"
    limit = max(1, int(os.getenv("LOGIN_RATE_LIMIT", "10")))
    window = max(1, int(os.getenv("LOGIN_RATE_WINDOW_SECONDS", "60")))
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis is not None and redis_url:
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            count = client.incr(key)
            if count == 1:
                client.expire(key, window)
            if count > limit:
                raise HTTPException(status_code=429, detail="too many login attempts")
            return
        except HTTPException:
            raise
        except Exception:
            pass
    count, expires_at = _local_rate_limits.get(key, (0, 0.0))
    now = time.time()
    if expires_at <= now:
        count, expires_at = 0, now + window
    count += 1
    _local_rate_limits[key] = (count, expires_at)
    if count > limit:
        raise HTTPException(status_code=429, detail="too many login attempts")
