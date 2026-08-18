# Corrected Dataset Ranking Diagnostics

Ranked members: 1,224
Events analyzed: 51
Date range: 2025-04-04 to 2026-03-29

## Highest Correlations With JPAR

- `mean_adjusted_event_jpar`: 0.994
- `trueskill_conservative`: 0.882
- `best3_mean_log_zscore`: 0.801
- `conservative_log_zscore`: 0.797
- `mean_event_percentile`: 0.793
- `mean_normalized_rank`: 0.786
- `robust_log_zscore`: 0.786
- `elo_rating`: 0.775
- `mean_log_zscore`: 0.774
- `weighted_event_percentile`: 0.772

## Notes

- `weighted_log_zscore`, `weighted_event_percentile`, Elo, and TrueSkill are useful comparison systems because they reduce or avoid direct dependence on JPAR's event calibration multiplier.
- Positive `rank_delta_vs_jpar` in the disagreement CSV means the alternative system ranks someone worse than JPAR; negative means it ranks them better.
- The PDF is the primary review artifact; CSVs are included only for drill-down.
