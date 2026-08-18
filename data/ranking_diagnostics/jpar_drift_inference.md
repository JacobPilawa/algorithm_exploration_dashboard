# JPAR Drift Inference

As-of median JPAR moved from `1.1380` to `1.4545` across the corrected cutoff run, a change of `+0.3165`.

## Mechanism

JPAR is not anchored to an invariant external scale. Each event first computes `event_jpar = completion_seconds / event_mean_completion_seconds`, then sometimes recalibrates that event using returning players' prior JPARs: `adjusted_event_jpar = completion_seconds / mean_expected_event_average`. Finally, a person's score is updated as a half-average of previous JPAR and current adjusted event JPAR.

This means the system is path-dependent. If the population entering the ranking changes, or if returning players at an event are not representative of the event field, the calibration factor can move the entire event up or down. New players inherit that shifted event scale as their initial JPAR, and then they become future calibration anchors. That feedback loop is the root reason drift can persist.

## Diagnostics From This Run

- Correlation between event calibration multiplier and new-row share: `0.310`.
- Compare the PDF drift decomposition page: if the fixed early cohort is stable while the all-ranked median drifts, population mix is a major driver. If the fixed cohort also drifts, the update/calibration mechanics are changing scores for existing people.
- Large one-event disagreements indicate that current JPAR can be strongly affected by a person's first event context, especially before enough history accumulates.

## Mitigation Options

1. Use log-z or percentile event scores as the primary event-normalized input. These are anchored to within-event distributions rather than raw event average ratios.
2. Keep the running update, but update toward a stable within-event metric (`weighted_log_zscore` or `weighted_event_percentile`) instead of recalibrated JPAR ratios.
3. If preserving JPAR, constrain calibration multipliers: shrink event calibration toward 1.0 unless there are enough returning anchors and their prior ratings are representative.
4. Add uncertainty or minimum-event rules for publication: provisional until 2 or 3 events; use conservative rank for one-event participants.
5. Publish both a performance rating and an uncertainty/experience field. This prevents one-event stars and one-event poor showings from being overinterpreted.
