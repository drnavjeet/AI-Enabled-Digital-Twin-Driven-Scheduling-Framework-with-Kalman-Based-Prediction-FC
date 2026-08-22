# Reviewer response additions

## Comment 16: packet loss and jitter

We agree that the original network model did not explicitly represent packet loss or jitter. The revised simulator now models time-varying uplink/downlink loss and jitter, retry-capped packet retransmissions, retransmission delay and energy, transport failure, and one-step QoS prediction. All schedulers experience identical paired QoS traces. Because the source datasets do not provide aligned loss/jitter observations, the revision uses three documented synthetic regimes and reports a separate one-factor-at-a-time sensitivity analysis. The manuscript now states this limitation explicitly.

## Comment 17: recent AI scheduler

We agree that the original baselines did not include a sufficiently recent learning scheduler. We added a paper-aligned adaptation of Wang and Sun's 2025 DRL scheduler with ordinal optimization (EURASIP Journal on Wireless Communications and Networking, DOI: 10.1186/s13638-025-02534-0). The implementation uses actor and critic networks, replay learning, target-network soft updates, binary action perturbations, 100 ordinal candidates, and critic top-10 filtering. Five independently seeded policies are trained only on training scenarios, frozen, and assigned evenly across the 30 paired evaluation seeds. Since the source article does not publish code or training data and our common simulator exposes venue selection rather than server-plus-VM allocation, we label this a paper-aligned discrete-action adaptation rather than an exact reproduction.
