# Licenses and attribution

BerryTrack software is AGPL-3.0-or-later; see the root LICENSE. The runtime uses
Ultralytics' AGPL-licensed implementation and model loading/training tools.
The original ByteTrack/BoT-SORT repositories are research references; the actual
adapters call the locked Ultralytics distribution. DeepSORT is supplied through
deep-sort-realtime (MIT); TrackEval uses its upstream MIT license. PyTorch is
BSD-style, OpenCV is Apache-2.0, NumPy/SciPy are BSD-style and Gradio is Apache-2.0.
Retain the installed distributions' license notices.

Data licenses do not become the software license:

- BUTom-ST21 is CC-BY-4.0. The only source-image derivatives tracked here are
  small failure illustrations with attribution in [assets/README.md](assets/README.md).
  Downloaded sequences and generated demo videos remain ignored.
- BlueberryDCM is CC-BY-NC-4.0. Data and adaptation weights are not committed.
  Four-class labels derive from VIA; format discrepancies are documented.
- AgriVision's shared Figshare record displays CC BY. The article's publication
  license must not be substituted for the data terms.
- NeSTCount's Zenodo record is CC-BY-4.0. Original images were not obtained or
  fabricated; the record directs users to request them from the authors.

Cached pretrained Ultralytics weights retain upstream terms and provenance.
Training with restricted data does not erase its restrictions. This repository
does not relicense data, source imagery or third-party model artifacts.
