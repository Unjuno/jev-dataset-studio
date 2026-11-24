"""Server-owned configuration. No credential is returned by the API."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os
import re


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    database_url: str = ""
    access_tokens: dict[str, str] = field(default_factory=dict, repr=False)
    jev_api_key: str = field(default="", repr=False)
    jev_model: str = "jev-1.14.0"
    verifier_api_key: str = field(default="", repr=False)
    verifier_base_url: str = ""
    verifier_model: str = ""
    input_price_per_million: float = 0.042
    verifier_price_per_million: float = 0.0
    max_upload_bytes: int = 20 * 1024 * 1024
    max_rows: int = 25000
    max_text_chars: int = 100000
    lease_seconds: int = 180
    max_attempts: int = 3
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]", "testserver", "web")
    secure_cookies: bool = False

    def __post_init__(self):
        path = Path(self.data_dir).resolve()
        object.__setattr__(self, "data_dir", path)
        if not self.database_url:
            object.__setattr__(self, "database_url", "sqlite:///" + str(path / "studio.sqlite3"))
        for workspace, token in self.access_tokens.items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", workspace):
                raise ValueError("Workspace names must use letters, digits, '_' or '-'.")
            if not isinstance(token, str) or len(token) < 24:
                raise ValueError("Each access token must contain at least 24 characters.")
        if len(set(self.access_tokens.values())) != len(self.access_tokens):
            raise ValueError("Access tokens must be unique per workspace.")
        if self.verifier_base_url and not self.verifier_base_url.startswith("https://"):
            raise ValueError("VERIFIER_BASE_URL must use HTTPS.")
        if any(not math.isfinite(p) or p < 0 for p in (self.input_price_per_million, self.verifier_price_per_million)):
            raise ValueError("Provider price estimates must be finite and nonnegative.")

    @property
    def verifier_ready(self) -> bool:
        return bool(self.verifier_api_key and self.verifier_base_url and self.verifier_model)

    @classmethod
    def from_env(cls):
        tokens = json.loads(os.environ.get("STUDIO_ACCESS_TOKENS", "{}"))
        if not isinstance(tokens, dict):
            raise ValueError("STUDIO_ACCESS_TOKENS must be a JSON object.")
        return cls(
            data_dir=Path(os.environ.get("STUDIO_DATA_DIR", "data")),
            database_url=os.environ.get("DATABASE_URL", ""),
            access_tokens=tokens,
            jev_api_key=os.environ.get("TYPESAFE_API_KEY", ""),
            jev_model=os.environ.get("JEV_MODEL", "jev-1.14.0"),
            verifier_api_key=os.environ.get("VERIFIER_API_KEY", ""),
            verifier_base_url=os.environ.get("VERIFIER_BASE_URL", ""),
            verifier_model=os.environ.get("VERIFIER_MODEL", ""),
            input_price_per_million=float(os.environ.get("JEV_INPUT_PRICE_PER_MILLION", "0.042")),
            verifier_price_per_million=float(os.environ.get("VERIFIER_PRICE_PER_MILLION", "0")),
            allowed_hosts=tuple(x.strip() for x in os.environ.get(
                "STUDIO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver,web"
            ).split(",") if x.strip()),
            secure_cookies=os.environ.get("STUDIO_SECURE_COOKIES", "false").lower() == "true",
        )
