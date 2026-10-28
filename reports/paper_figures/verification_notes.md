# Verification notes

- Rendered stems: fig02_postprocessing_group_error_rates, fig03_adversarial_group_error_rates, fig04_postprocessing_target_eod_ba, fig05_adversarial_six_units, fig06_rq4b_secondary_deltas.
- Formats: PDF, SVG, PNG at 300 DPI.
- Internal titles and panel labels: none.
- Figure 2 markers: circle before, square after; connectors are within group only.
- Figure 3 markers: circle frozen, square retrained control, diamond adversarial; no lines join groups.
- Figure 2 has no seed-variability interval; Figure 3 error bars are sample SD across seeds 42--44 with ddof=1, not confidence intervals.
- RQ4b Figure 5 is restricted to seeds 42/43/44 and lambda 0/0.01.
- Figure 6 contains matched delta EOD6, DPD3, and DPD6; DPD deltas have no inferred CI.
- Frozen artifacts were read only; no fit, inference, threshold selection, Test re-evaluation, or bootstrap generation occurred.
- Rerender manually from the notebook; this notebook is not auto-executed by the implementation task.
