-- One-line overview of a log.
SELECT
    COUNT(*) AS n_queries,
    AVG(r.clicks) AS mean_clicks,
    MIN(r.propensity) AS min_propensity,
    MAX(r.propensity) AS max_propensity,
    SUM(CASE WHEN r.clicks > 0 THEN 1 ELSE 0 END) AS queries_with_click
FROM (
    SELECT q.row_idx, q.propensity, COALESCE(SUM(i.clicked), 0) AS clicks
    FROM queries AS q
    LEFT JOIN impressions AS i ON i.row_idx = q.row_idx
    GROUP BY q.row_idx
) AS r;
