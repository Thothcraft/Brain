# Radio localization v1: model and implementation boundary

## Selected formulation

The target design is the cooperative Bayesian factor-graph formulation in
Wymeersch, Lien, and Win (2009), *Cooperative Localization in Wireless
Networks*, Proceedings of the IEEE 97(2), 427–450,
<https://doi.org/10.1109/JPROC.2008.2008853>. The graph has position variables
for participating physical devices and factors for manual anchors, radio
measurements, motion, confirmed co-location, sufficiently accurate GPS,
building-boundary evidence, and verified device identity. Radios on the same
physical host share one position variable and therefore do not create
independent geometric information.

For a radio measurement from physical scanner `i` to transmitter `j`, the
measurement factor uses a calibrated log-distance likelihood:

`r_ij ~ StudentT(nu, P0_i,c - 10 n_c log10(||x_i-x_j||/d0), sigma_i,c)`

where `c` identifies radio family and channel/band. Student-t tails reduce the
effect of multipath and interference outliers. `P0`, path-loss exponent `n`,
receiver offset, and variance must come from user configuration or sufficient
calibration data; they are not universal BLE/Wi-Fi constants. CSI is not a
separate range measurement unless a validated ranging feature is available.

The intended inference is robust particle belief propagation: particles retain
multiple position modes, radio factors reweight particle pairs, and anchor,
motion, GPS, and building factors update the same beliefs. A bounded particle
set is an approximation; its particle count, resampling, and credible-region
coverage must be reported and calibrated on held-out sessions. Before emitting
a point summary, compute the local information matrix from the active factors.
If geometry is rank deficient or ill-conditioned, publish the posterior locus
and an `unobservable` quality state rather than a coordinate. A single range
factor is radial; two can remain symmetric; collinear or colocated scanners
remain poorly conditioned. Altitude is only estimated when independent
vertical evidence exists.

## Publication boundary

The UI currently publishes radio edges from authenticated, account-scoped
context evidence. The map uses existing room placements for placed device
coordinates and leaves other devices unlocated. RSSI edges do not provide
bearing, and display proximity must not be described as a measured distance.
Building membership remains indeterminate without usable footprint geometry
and a location posterior. The existing `room/v1` placement data is retained;
this design adds no identity registry or storage system.

## Required adaptations and assumptions

- Preserve observation timestamp, observer host and radio/interface identity,
  modality, frequency/channel when supplied, RSSI, optional Tx power, and
  source-quality metadata through a bounded live aggregation path. Database
  evidence sampling is not a suitable estimator input stream.
- Join radios across scanners only using a trustworthy stable identifier or an
  account-scoped pseudonymous correlation key. Rotating identifiers remain
  ephemeral; names and manufacturer data are not identity evidence.
- Treat manual scanner placement as an uncertain, expiring observation.
  Movement invalidates or weakens that factor. Mobile scanners require their
  own position belief.
- A common-room calibration constrains devices to a shared area, not one exact
  point. Only explicit close co-location at a known position can support that
  stronger factor. One-room data does not establish an empirical radio map.
- Site coordinates alone cannot establish a building boundary. Missing
  footprint means inside/outside is indeterminate.
- No consented multi-account sharing is implemented by this version. Future
  cooperative inputs require explicit per-site permission and must not expose
  contributors' identities or movement history.

## Current implementation status

The Flutter destination is consolidated to the existing evidence map. It
retains actual radio links, room placements, and device detail selection, and
shows unknown transmitters by default. The probabilistic estimator, live
high-rate observation buffer, calibration workflow, geographic building
overlay, map history, and multi-user sharing described above are design
requirements, not implemented behavior. Do not treat the old canvas
RSSI-to-distance solver as validated localization; it is no longer selected
as the map destination and should be removed when its remaining consumers are
confirmed.

## References

- H. Wymeersch, J. Lien, and M. Z. Win, “Cooperative Localization in Wireless
  Networks,” *Proceedings of the IEEE*, 97(2), 427–450, 2009.
- H. Ferris, D. Hähnel, and D. Fox, “Gaussian Processes for Signal
  Strength-Based Location Estimation,” *Robotics: Science and Systems II*,
  2006, <https://doi.org/10.15607/RSS.2006.II.039>.

The Gaussian-process approach is a possible learned likelihood only after
adequate position-labeled calibration across the relevant spatial extent; a
single-room session is insufficient.
