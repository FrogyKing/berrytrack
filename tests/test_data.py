import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from berrytrack.data.annotation import create_annotation_task, export_verified_task
from berrytrack.data.download import download, extract_archive, file_digest, safe_archive_path
from berrytrack.data.loaders import grouped_split
from berrytrack.data.mot import read_mot, validate_splits, write_mot
from berrytrack.data.public import ArrayUnpickler, verify_manifest_files


def test_mot_round_trip_and_missing_gt(tmp_path):
    path = tmp_path / "gt.txt"
    with pytest.raises(FileNotFoundError):
        read_mot(path)
    rows = np.array([[2, 10, 1, 1, 3, 4, 1, 1, 1, -1], [1, 10, 1, 1, 3, 4, 1, 1, 1, -1]])
    write_mot(path, rows)
    assert read_mot(path)[:, 0].tolist() == [1, 2]
    path.write_text("1,1,1,1,-3,4,1\n")
    with pytest.raises(ValueError, match="positive"):
        read_mot(path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/windows/file", "a/../../b"])
def test_archive_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        safe_archive_path(tmp_path, name)


def test_archive_symlink_and_budget(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as out:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../target"
        out.addfile(info)
    with pytest.raises(ValueError, match="regular"):
        extract_archive(archive, tmp_path / "output", 10)
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("a", "long contents")
    with pytest.raises(ValueError, match="budget"):
        extract_archive(archive, tmp_path / "output", 2)


def test_verified_download(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"real file")
    digest = file_digest(source)
    target = download(source.as_uri(), tmp_path / "target", "sha256:" + digest, 50)
    assert target.read_bytes() == b"real file"
    with pytest.raises(ValueError, match="Checksum"):
        download(source.as_uri(), tmp_path / "wrong", "sha256:" + "0" * 64, 50)
    assert not (tmp_path / "wrong.part").exists()


def test_no_split_leakage():
    with pytest.raises(ValueError):
        validate_splits({"train": ["a"], "test": ["a"]})
    split = grouped_split({"plant1_frame1": "plant1", "plant1_frame2": "plant1"})
    assert sorted(map(len, split.values())) == [0, 0, 2]


def test_manual_verification_required(tmp_path):
    task = tmp_path / "task.json"
    create_annotation_task(Path("video.mp4"), task, 2, "CC-BY-4.0")
    with pytest.raises(ValueError, match="verification"):
        export_verified_task(task, tmp_path / "gt.txt")
    data = json.loads(task.read_text())
    data.update(label_kind="manual_verified", reviewer="Human reviewer")
    for frame in data["frames"]:
        frame["reviewed"] = True
        frame["objects"] = [{"track_id": 1, "bbox_xywh": [0, 1, 5, 6]}]
    task.write_text(json.dumps(data))
    export_verified_task(task, tmp_path / "gt.txt")
    assert read_mot(tmp_path / "gt.txt")[:, 2].tolist() == [1, 1]


def test_unpickler_rejects_executable_globals():
    payload = pickle.dumps(Path("forbidden"))
    with pytest.raises(pickle.UnpicklingError, match="Disallowed"):
        ArrayUnpickler(io.BytesIO(payload)).load()


def test_unpickler_allows_numpy_legacy_array():
    array = np.arange(4)
    payload = pickle.dumps(array, protocol=4)
    assert np.array_equal(ArrayUnpickler(io.BytesIO(payload)).load(), array)


def test_mot_six_column_confidence_is_explicitly_one(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,1,1,1,5,5\n")
    assert read_mot(path)[0, 6] == 1


def test_cached_data_digest_rejects_corruption(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"valid bytes")
    entries = [{"path": "frame.jpg", "sha256": file_digest(path)}]
    verify_manifest_files(tmp_path, entries)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_manifest_files(tmp_path, entries)
