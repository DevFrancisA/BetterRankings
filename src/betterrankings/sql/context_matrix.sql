-- Context features in row-major order, ready to reshape to (queries, dims).
SELECT row_idx, dim, value
FROM contexts
ORDER BY row_idx, dim;
