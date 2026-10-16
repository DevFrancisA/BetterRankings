-- Click table by position. Under the logging policy the click rate falls with
-- rank partly because of position bias, not only because worse items sit lower.
SELECT
    position,
    COUNT(*) AS impressions,
    SUM(clicked) AS clicks,
    AVG(CAST(clicked AS REAL)) AS click_rate
FROM impressions
GROUP BY position
ORDER BY position;
