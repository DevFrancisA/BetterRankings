-- Shown items in row-major order, ready to reshape to (queries, positions).
SELECT row_idx, position, item_id, pop, clicked
FROM impressions
ORDER BY row_idx, position;
