-- Total clicks per logged query, in file order. This is the reward every
-- estimator averages.
SELECT
    q.row_idx,
    q.query_id,
    q.propensity,
    q.target_propensity,
    COALESCE(SUM(i.clicked), 0) AS clicks,
    COUNT(i.position) AS n_shown
FROM queries AS q
LEFT JOIN impressions AS i ON i.row_idx = q.row_idx
GROUP BY q.row_idx
ORDER BY q.row_idx;
