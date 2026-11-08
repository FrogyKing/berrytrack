import json

import pytest
from PIL import Image

from berrytrack.detection.training import CLASS_NAMES, prepare_detection_split, via_labels


def test_via_class_conversion_preserves_occlusion_and_ripeness(tmp_path):
    image = tmp_path / "fruit.jpg"
    Image.new("RGB", (100, 100)).save(image)
    shapes = [{"name": "rect", "x": 10, "y": 20, "width": 10, "height": 10}] * 4
    attrs = [{"Blueberry": {name: True}} for name in CLASS_NAMES]
    image.with_suffix(".json").write_text(
        json.dumps(
            {
                "key": {
                    "filename": image.name,
                    "regions": {"shape_attributes": shapes, "region_attributes": attrs},
                }
            }
        )
    )
    labels, counts = via_labels(image)
    assert [int(line.split()[0]) for line in labels.splitlines()] == [0, 1, 2, 3]
    assert list(counts.values()) == [1, 1, 1, 1]
    assert all(
        line.split()[1:] == ["0.15000000", "0.25000000", "0.10000000", "0.10000000"]
        for line in labels.splitlines()
    )


def test_preparing_a_smaller_split_refuses_stale_training_files(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for i in range(10):
        image = root / f"fruit{i}.jpg"
        Image.new("RGB", (20, 20)).save(image)
        image.with_suffix(".txt").write_text("0 .5 .5 .2 .2\n")
        image.with_suffix(".json").write_text(
            json.dumps(
                {
                    "key": {
                        "filename": image.name,
                        "regions": {"shape_attributes": [], "region_attributes": []},
                    }
                }
            )
        )
    output = tmp_path / "split"
    prepare_detection_split(root, output, max_train=7)
    with pytest.raises(ValueError, match="different split"):
        prepare_detection_split(root, output, max_train=2)
