# Visual asset attribution

`agrivision-reference.jpg` shows eight real AgriVision DB-2 crops with56 boxes
and7 identities authored by assistant visual inspection. This is a tiny,
near-stationary exploratory reference, not human-verified or independent GT.
Source: [AgriVision v1](https://doi.org/10.6084/m9.figshare.29222462.v1).
Authors: Muhammad Owais, Muhammad Shafay, Muhammad Zubair, Shamal Mohammed,
Lakmal Seneviratne and Irfan Hussain. License: CC-BY-4.0 per dataset metadata.
Modifications: cropped source frames1–8, added identity boxes/frame labels,
and assembled a contact sheet. No endorsement by the source authors is implied.
The article has separate licensing; the source dataset's license is used here.

`tracking-demo.gif` is a three-second, reduced-resolution derivative of the real
BUTom-ST21 sequence235 demo. It displays BerryTrack and BoT-SORT outputs side by
side from identical reference detections, with persistent IDs, trajectories and
confirmed unique-ID counts. Frames were encoded as H.264 and then an animated GIF.
It is a software demonstration with remaining identity errors, not detector
performance or a raw dataset redistribution. The same author attribution, DOI
and CC-BY-4.0 license below apply.

`identity-switches.png` is a derived visualization of BUTom-ST21 sequence 235.
The source dataset is by **Michael Halstead, Esra Guclu, Mohamed Farag, Enrico
Pallotta, Christian Hund, Ribana Roscher, Maren Bennewitz, Juergen Gall, and Chris
McCool**, as listed in the publisher's dataset metadata.

- Source: [BUTom-ST21 v1](https://doi.org/10.60507/FK2/TTPCNV).
- Source license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Modifications: cropped and resized source images, overlaid human-corrected GT
  and BerryTrack predictions, and added frame identifiers, source timestamps,
  before/after labels and measured switch descriptions.
- Provenance: [identity-switches.json](identity-switches.json).

These qualitative examples show observed failures in an oracle-box development
diagnostic. They do not imply endorsement by the dataset authors or a validated
blueberry tracking benchmark. No raw dataset archive, original image sequence,
model weights or annotation dump is included in this directory.
