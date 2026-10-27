"""API-key auth. /health stays public; everything else requires a key.

Accepted (per the AML contract):
  Authorization: Bearer <key>
  Authorization: Token <key>
  X-Api-Key: <key>

The key comes from Settings (config.py); make_require_api_key builds the
FastAPI dependency bound to that key.
"""
from typing import Callable, Optional

from fastapi import Header, HTTPException


def make_require_api_key(api_key: str) -> Callable:
    def require_api_key(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ) -> bool:
        if x_api_key and x_api_key == api_key:
            return True
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() in ("bearer", "token") and token == api_key:
                return True
        raise HTTPException(status_code=401, detail="invalid api key")
    return require_api_key
