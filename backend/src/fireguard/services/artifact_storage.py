from pathlib import Path

from fireguard.core.errors import APIError


class LocalArtifactStorage:
    """Resolve relative artifact URIs beneath one configured local root."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, storage_uri: str) -> Path:
        candidate = Path(storage_uri)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise APIError(400, "INVALID_ARTIFACT_URI", "Artifact URI must be a safe relative path")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise APIError(
                400, "INVALID_ARTIFACT_URI",
                "Artifact URI escapes the configured storage root",
            )
        if not resolved.is_file():
            raise APIError(
                404, "ARTIFACT_NOT_FOUND",
                "The requested static artifact is not available",
            )
        return resolved
