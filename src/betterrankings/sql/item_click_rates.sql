-- Raw click rate per item, ignoring position. Items the logging policy put at
-- the top look better here than they are.
SELECT
    item_id,
    COUNT(*) AS impressions,
    SUM(clicked) AS clicks,
    AVG(CAST(clicked AS REAL)) AS click_rate,
    AVG(CAST(position AS REAL)) AS mean_position
FROM impressions
GROUP BY item_id
HAVING COUNT(*) >= :min_impressions
ORDER BY click_rate DESC, impressions DESC, item_id
LIMIT :limit;
