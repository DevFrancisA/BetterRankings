-- Item features in row-major order, ready to reshape to (queries, positions, dims).
SELECT row_idx, position, dim, value
FROM item_features
ORDER BY row_idx, position, dim;
