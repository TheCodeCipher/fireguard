from __future__ import annotations

import hashlib
from pathlib import Path

from fireguard.core.errors import APIError


class LocalArtifactStorage:
    """Resolve and verify relative artifact URIs beneath one configured root."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, storage_uri: str) -> Path:
        if not storage_uri or "\\" in storage_uri:
            raise APIError(
                400, "INVALID_ARTIFACT_URI", "Artifact URI must use safe relative POSIX paths"
            )

        candidate = Path(storage_uri)
        if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
            raise APIError(400, "INVALID_ARTIFACT_URI", "Artifact URI must be a safe relative path")

        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise APIError(
                400, "INVALID_ARTIFACT_URI", "Artifact URI escapes the configured storage root"
            )
        if not resolved.is_file():
            raise APIError(
                404, "ARTIFACT_NOT_FOUND", "The requested static artifact is not available"
            )
        return resolved

    @staticmethod
    def checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def inspect(self, storage_uri: str) -> dict[str, int | str]:
        path = self.resolve(storage_uri)
        return {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "sha256": self.checksum(path),
        }

    def verify(
        self, storage_uri: str, expected_sha256: str | None, expected_size: int | None = None
    ) -> Path:
        path = self.resolve(storage_uri)
        if not expected_sha256:
            raise APIError(
                409, "ARTIFACT_CHECKSUM_MISSING", "Published artifact has no SHA-256 checksum"
            )
        actual_sha256 = self.checksum(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise APIError(
                409,
                "ARTIFACT_CHECKSUM_MISMATCH",
                "Artifact checksum does not match its registry record",
            )
        if expected_size is not None and path.stat().st_size != expected_size:
            raise APIError(
                409, "ARTIFACT_SIZE_MISMATCH", "Artifact size does not match its registry record"
            )
        return path
