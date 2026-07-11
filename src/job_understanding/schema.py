"""
Core job posting schema — deliberately minimal and provider-agnostic.

`Job` holds only what's true of a job posting regardless of source
(LinkedIn, Greenhouse, Lever, ...) — no application-tracking state, no
user-specific fields. Anything downstream (candidate matching, tracking,
persistence) is a separate concern layered on top of this, not baked in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = ["Job"]


@dataclass
class Job:
    # --- Core job data (scraping/ingestion output only) ---
    job_id: str
    title: str
    company: str
    location: str
    url: str
    source: str  # e.g. "linkedin", "greenhouse", "lever"

    description: str = ""
    posted_at: Optional[datetime] = None
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        if not self.url:
            raise ValueError("url cannot be empty")

    def __repr__(self) -> str:
        return f"Job(id={self.job_id!r}, title={self.title!r}, company={self.company!r})"
