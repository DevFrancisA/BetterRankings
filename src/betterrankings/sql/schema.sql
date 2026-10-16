-- Log layer: one wide CSV row per query is split into long tables so the
-- aggregations below are plain GROUP BY queries.
CREATE TABLE queries (
    row_idx INTEGER PRIMARY KEY,          -- file order, 0-based
    query_id TEXT NOT NULL,
    propensity REAL NOT NULL,             -- logging probability of the shown ranking
    target_propensity REAL                -- optional, supplied by the user
);

CREATE TABLE contexts (
    row_idx INTEGER NOT NULL REFERENCES queries(row_idx),
    dim INTEGER NOT NULL,                 -- 1-based feature number
    value REAL NOT NULL,
    PRIMARY KEY (row_idx, dim)
);

CREATE TABLE impressions (
    row_idx INTEGER NOT NULL REFERENCES queries(row_idx),
    position INTEGER NOT NULL,            -- 1 is the top of the list
    item_id TEXT NOT NULL,
    pop REAL NOT NULL,
    clicked INTEGER NOT NULL CHECK (clicked IN (0, 1)),
    PRIMARY KEY (row_idx, position)
);

CREATE TABLE item_features (
    row_idx INTEGER NOT NULL,
    position INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (row_idx, position, dim),
    FOREIGN KEY (row_idx, position) REFERENCES impressions(row_idx, position)
);
