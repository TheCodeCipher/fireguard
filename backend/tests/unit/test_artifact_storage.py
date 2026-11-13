import pytest

from fireguard.core.errors import APIError
from fireguard.services.artifact_storage import LocalArtifactStorage


def test_storage_resolves_relative_artifact(tmp_path):
    artifact = tmp_path / "mock" / "map.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    assert LocalArtifactStorage(tmp_path).resolve("mock/map.json") == artifact.resolve()


@pytest.mark.parametrize("uri", ["../secret.txt", "../../etc/passwd", "/etc/passwd"])
def test_storage_rejects_absolute_and_traversal(uri, tmp_path):
    with pytest.raises(APIError) as exc:
        LocalArtifactStorage(tmp_path).resolve(uri)
    assert exc.value.code == "INVALID_ARTIFACT_URI"
