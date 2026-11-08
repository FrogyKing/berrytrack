import io
import zipfile

import numpy as np
import pytest

from berrytrack.data.agrivision import (
    RangeReader,
    mask_component_boxes,
    prepare_agrivision,
    prepare_assistant_reference,
    select_frames,
)


def test_mask_proposals_preserve_merged_regions_instead_of_inventing_fruit_instances():
    mask = np.zeros((40, 50), np.uint8)
    mask[2:12, 3:13] = 255
    mask[6:16, 10:20] = 255  # Touching fruit are one inferred connected region.
    mask[22:32, 30:40] = 255
    mask[38, 48] = 255  # Isolated segmentation speck is below minimum area.
    boxes = mask_component_boxes(mask)
    np.testing.assert_array_equal(boxes, [[3, 2, 20, 16], [30, 22, 40, 32]])


def test_mask_proposals_reject_color_masks_and_handle_empty_foreground():
    assert mask_component_boxes(np.zeros((30, 30), np.uint8)).shape == (0, 4)
    with pytest.raises(ValueError, match="2D mask"):
        mask_component_boxes(np.zeros((30, 30, 3), np.uint8))


def test_range_reader_handles_zip_seeking_and_crc():
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("frame.jpg", b"test image")
    data = source.getvalue()
    reader = RangeReader("unused", len(data), max_bytes=2000, fetch=lambda a, b: data[a : b + 1])
    with zipfile.ZipFile(reader) as archive:
        assert archive.read("frame.jpg") == b"test image"
    assert reader.bytes_fetched < 2000


def test_range_reader_enforces_total_transfer_budget():
    data = bytes(range(100))
    reader = RangeReader("unused", len(data), max_bytes=8, fetch=lambda a, b: data[a : b + 1])
    assert reader.read(4) == data[:4]
    assert reader.read(4) == data[4:8]
    with pytest.raises(ValueError, match="budget"):
        reader.read(1)


def test_server_ignoring_range_is_rejected_before_body_download(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Length": "78036031533"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            pytest.fail("A full-archive response body must never be read")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    reader = RangeReader("https://example.org/archive.zip", 100)
    with pytest.raises(ValueError, match="honor"):
        reader.read(10)


def test_numeric_frame_order_and_missing_source_frame():
    prefix = "DB-2/Camera 1/Videos_Frames/Camera 1 Video (1)_"
    names = [prefix + str(i) + ".jpg" for i in (1, 10, 11, 2, 3)]
    assert select_frames(names, 1, 1, 1, 3) == [prefix + str(i) + ".jpg" for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="contiguous"):
        select_frames(names, 1, 1, 1, 4)


@pytest.mark.parametrize(
    "camera,video,start,count",
    [(0, 1, 1, 2), (1, 0, 1, 2), (1, 1, 0, 2), (1, 1, 1, 41), (1, 1, 1, 0)],
)
def test_invalid_selection_rejected_without_network(camera, video, start, count, tmp_path):
    with pytest.raises(ValueError):
        prepare_agrivision(tmp_path / "output", camera, video, start, count)


def test_existing_output_not_mixed_with_new_sequence(tmp_path):
    (tmp_path / "manifest.json").write_text("existing")
    with pytest.raises(FileExistsError):
        prepare_agrivision(tmp_path)


def test_visual_reference_cannot_be_applied_to_uninspected_frames(tmp_path):
    (tmp_path / "img1").mkdir()
    (tmp_path / "img1" / "000001.jpg").write_bytes(b"a different source image")
    with pytest.raises(ValueError, match="visually inspected"):
        prepare_assistant_reference(tmp_path, tmp_path / "reference")
    assert not (tmp_path / "reference").exists()
