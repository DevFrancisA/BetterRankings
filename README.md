# BetterRankings

BetterRankings estimates how a new ranking policy would perform from logged data alone, without running a live A/B test. It then checks those estimates against a simulated online test where the true answer is known.

**Live app:** https://devfrancisa.github.io/BetterRankings/

It ships as two things that compute the same numbers:

- a Python package, `betterrankings`, with a command line tool;
- a static web app (TypeScript, no server, no tracking) that runs the same simulator and estimators in a Web Worker in your browser.

All built-in data is simulated.

## The idea in brief

A ranking system shows each query a list of items, and people click some of them. Suppose the current policy shows lists with some randomness and records the probability it gave to each list it showed (the *propensity*). Then its logs can be reweighted to answer a different question: how many clicks would another policy have earned? This is off-policy evaluation.

The simulator knows every true click probability, so it can also run the online test exactly. That gives a known answer to judge each estimator against.

## Run it locally

### Python package

Requires Python 3.10 or newer.

```bash
pip install -e ".[test]"

# Write a simulated log
betterrankings simulate --queries 5000 --exploration 0.2 --seed 7 --out log.csv

# Estimate one policy (or --policy all) from a log CSV
betterrankings evaluate --log log.csv --policy relevance --clip 20 --json results.json

# Simulate, estimate every candidate and compare with the simulated online test
betterrankings compare --queries 5000 --exploration 0.2 --seed 7

# Check the estimators on real logs (see "Real data" below)
betterrankings obd --data open_bandit_dataset --campaign every

pytest
```

Policies: `current`, `popularity`, `relevance`, `relevance_soft`, `noisy_logging`, `reversed`. Use `target_propensity` to evaluate the probabilities in your CSV's `target_propensity` column. `--clip none` turns clipping off.

The package is also importable:

```python
from betterrankings import run_comparison, evaluate_frame, read_log_csv

cmp = run_comparison(n_queries=5000, exploration=0.2, seed=7)
print(cmp.table())               # each estimator next to the true value
print(cmp.rank_agreement())      # policy pairs each estimator orders wrongly

ev = evaluate_frame(read_log_csv("log.csv"), ["relevance"], clip=20)
print(ev.policies["relevance"].estimates["doubly_robust"])
```

### Web app

Requires Node 20.19 or newer (22.12+ for the tests).

```bash
cd web
npm install
npm run dev       # http://localhost:5173/BetterRankings/
npm test          # unit tests plus parity checks against the Python fixtures
npm run build     # static site in web/dist
```

The web app has five parts:

- **Playground:** set the logging policy's exploration, the number of logged queries and a seed, pick a candidate, and run.
- **Results:** each estimator's estimate and 95% interval on a shared axis with the true online value. A clip slider updates clipped IPS and doubly robust live.
- **Ranking:** all candidates ranked by each estimator against the true order, with the candidates worth a live test marked.
- **Diagnostics:** the weight distribution, effective sample size, plain-language overlap warnings, and clicks by position.
- **Your log:** upload a CSV and evaluate a chosen target policy. The file is parsed in the browser and never leaves your device.

## Log CSV format

One row per query. Positions count from 1 at the top of the list, and lists may hold 2 to 6 items. A sample file is at [`web/public/sample-log.csv`](web/public/sample-log.csv), and the site offers it as a download.

| Column | Meaning |
| --- | --- |
| `query_id` | Any identifier |
| `ctx_1` .. `ctx_3` | Query context features |
| `item_K` | Id of the item shown at position K |
| `click_K` | 1 if that item was clicked, else 0 |
| `item_K_pop` | Popularity feature of that item |
| `item_K_f1` .. `item_K_f3` | Other item features |
| `propensity` | Probability the logging policy gave the whole ranking that was shown (greater than 0, at most 1) |
| `target_propensity` | Optional. Probability your own target policy gives the shown ranking |

The shown items are treated as the candidate set: a target policy re-ranks exactly those items. The built-in policies score items with fixed weights from the simulator, so they expect 3 context features and 3 item features. To evaluate your own policy, supply `target_propensity`. The reward model and doubly robust estimators need the target's probability of every possible ranking, so they are not available in that mode.

## The simulator

- **Catalog:** 200 items. Each has a popularity score and 3 features.
- **Queries:** each query has 3 context features and 4 candidate items drawn from the catalog. A ranking orders all 4, so there are 24 possible rankings, and every one is enumerated exactly.
- **Click model:** a position-based model. Item *j* at position *k* is clicked with probability `theta_k * sigmoid(0.6 * pop_j + x' W v_j - 1)`, where `theta_k = 1 / k`. Attention falls with rank, and relevance depends on the query and the item.
- **Logging policy:** with probability `1 - epsilon`, Plackett-Luce over a weaker scoring model; otherwise a uniformly random ranking. `epsilon` is the exploration level. The log records the exact probability of the shown ranking.
- **Candidate policies:**

  | Policy | Behaviour |
  | --- | --- |
  | Popularity | Greedy sort by popularity |
  | Relevance model | Greedy sort by a good but imperfect relevance model |
  | Soft relevance | Plackett-Luce over the same model |
  | Noisier logging | More exploration and a flatter temperature |
  | Reversed relevance | Deliberately worse |
  | Current policy | The logging policy itself, as a sanity check |

  Each policy returns a probability for every ranking, so it can score any logged ranking.
- **Ground truth:** for fresh simulated traffic, the exact expected clicks per query of each policy. The expectation over rankings is computed in closed form, with no click sampling.
- **Seeds:** every random draw comes from a seeded counter-based generator (mulberry32) that both implementations share. The same seed gives the same log and the same answers in Python and in the browser.

## Estimators and their assumptions

Notation for logged query *i* with total clicks `r_i`:

| Symbol | Meaning |
| --- | --- |
| `mu_i` | Logging propensity of the shown ranking |
| `pi_i` | Target policy's probability of that ranking |
| `w_i = pi_i / mu_i` | Importance weight |
| `q_i` | Reward model's predicted clicks for the shown ranking |
| `d_i` | Reward model's expected clicks under the target policy |

All of the weighting estimators assume three things:

- **Correct propensities:** `mu_i` is the true probability the logging policy gave the shown ranking.
- **Support:** wherever the target policy can show a ranking, the logging policy could have shown it too.
- **Stable behaviour:** queries are independent and user behaviour does not depend on which policy is running.

| Estimator | Formula | Assumptions and behaviour |
| --- | --- | --- |
| Naive click rate | `sum(pi_i r_i) / sum(pi_i)` | Average clicks on logged queries where the target would have shown the same list. It ignores the logging propensities, so it assumes the old policy chose lists independently of how clickable they were. That is rarely true, so this is the baseline that misleads. |
| Inverse propensity (IPS) | `mean(w_i r_i)` | Unbiased under the shared assumptions. Variance grows with the size of the weights. |
| Self-normalised IPS | `sum(w_i r_i) / sum(w_i)` | Same assumptions as IPS. Slightly biased in small samples (the bias shrinks as 1/n), usually with much lower variance. |
| Clipped IPS | `mean(min(w_i, c) r_i)` | Caps each weight at the clip level `c`. Lower variance, but biased downward whenever weights exceed `c`. `c` is a parameter. |
| Reward model (direct method) | `mean(d_i)` | A ridge logistic regression predicts clicks from context, item features, their interactions and position, and is cross-fitted. It needs no propensities, but it is only as good as the model, and its bias does not shrink with more data. |
| Doubly robust | `mean(d_i + min(w_i, c)(r_i - q_i))` | Starts from the reward model and corrects it with weighted residuals. Without clipping it is unbiased if either the propensities or the reward model are correct. Clipping trades some of that guarantee for lower variance. |

The tests check four of these claims directly:

- IPS and doubly robust are unbiased on a case with a known answer.
- Clipping lowers variance on heavy-tailed weights.
- Doubly robust stays on target with a wrong reward model, and with wrong weights.
- Runs are reproducible from a seed.

**Cross-fitting.** Rows are split into 2 folds. The reward model that scores a row is trained only on the other fold, so no row is predicted by a model that saw its clicks.

**Confidence intervals.** These are 95% percentile bootstrap intervals over logged queries, with 200 resamples by default. The reward model is not refitted inside the bootstrap, so the intervals for the reward model and doubly robust leave out model-fitting uncertainty.

**Effective sample size.** This is `(sum w)^2 / sum w^2`: how many equally weighted queries the weights are worth. It is reported for each estimator using the weights that estimator actually uses (clipped where relevant).

**Overlap warnings.** Each warning has a code and a plain-language message:

- `no_support`: the target never picks a logged ranking.
- `poor_overlap`: the effective sample size is below 30 queries or below 1% of the rows, whichever is larger.
- `limited_overlap`: the effective sample size is below 10% of the rows.
- `extreme_weight`: one query carries more than 10% of the total weight.
- `weights_off`: the mean weight is far from 1, which points to poor coverage or wrong propensities.

## Real data: Open Bandit Dataset

The simulator grades the estimators against an answer of its own making. `betterrankings obd` checks them on real logs instead. The [Open Bandit Dataset](https://research.zozo.com/data.html) holds a week of live traffic from the ZOZOTOWN fashion site, where a uniform random policy and Bernoulli Thompson sampling (BTS) ran side by side, so each policy's real click rate is known.

The command runs two experiments for each campaign (all, men, women):

- **BTS from random logs:** estimate BTS's click rate from the random policy's logs, and compare with BTS's measured click rate. BTS's target probabilities come from its own logged behaviour; with `--prior`, its published production prior is evaluated too.
- **Random from BTS logs:** estimate the random policy's click rate from BTS's logs. Its weights are heavy, so this one exercises the overlap diagnostics.

How it adapts the estimators:

- The dataset logs one item per position, so the weights are per position: `target(item | position) / logging(item | position)`.
- The reward model is an additive logistic model over item, position, time and user features, cross-fitted.
- Bootstrap intervals resample whole recommendations, grouping rows logged within 20 ms of each other.

```bash
betterrankings obd --data open_bandit_dataset --campaign every --prior prior_bts.yaml --cache obd_cache
```

The download steps, the method's assumptions and the measured results are in [`results/OBD.md`](results/OBD.md). The tests run on a 3,000-row public sample in `tests/data/obd/`. The data is CC BY 4.0, from Saito et al., "Open Bandit Dataset and Pipeline" (NeurIPS Datasets and Benchmarks, 2021).

## Python and browser parity

`scripts/make_fixtures.py` writes the shared fixture files:

- simulated logs in `fixtures/*.csv`;
- the sample upload, `web/public/sample-log.csv`;
- the Python results in `fixtures/expected.json`: truth, estimates, intervals, effective sample sizes, warnings, and estimates at several clip levels.

Two test suites keep the implementations tied together. `tests/test_fixtures.py` fails if the committed fixtures drift from the Python code. `web/test/parity.test.ts` checks that the TypeScript port does three things:

- simulates the identical logs;
- runs the identical online test;
- reproduces every Python number to within 1e-9 relative.

## SQL log layer

In Python, a log CSV is validated and loaded into an in-memory SQLite database as long tables (`queries`, `contexts`, `impressions`, `item_features`). Per-query rewards, per-position click tables, item click rates and the log summary all come from the queries in [`src/betterrankings/sql/`](src/betterrankings/sql/).

## Repository layout

```
src/betterrankings/     Python package
  rng.py                shared seeded random numbers (mulberry32)
  world.py              simulator: catalog, queries, position-based click model
  policies.py           logging and candidate policies, ranking probabilities
  logs.py               simulated logs and the CSV format
  sqlstore.py, sql/     SQLite log layer and its queries
  reward_model.py       cross-fitted logistic click model
  estimators.py         estimators, bootstrap, diagnostics
  evaluate.py           evaluation pipeline and simulated online test
  obd.py                real-data checks on the Open Bandit Dataset
  cli.py                betterrankings simulate | evaluate | compare | obd
tests/                  pytest suite (tests/data/obd: public dataset sample)
results/                measured results on real data (OBD.md)
scripts/make_fixtures.py  writes the shared fixtures
fixtures/               logs and expected results shared with the web app
web/                    static web app (Vite + TypeScript)
  src/core/             TypeScript port of the simulator, policies and estimators
  src/worker.ts         Web Worker running all computation
  src/ui/               hand-built SVG charts
  test/                 Vitest unit and parity tests
.github/workflows/      CI (Python tests, TypeScript tests, build) and Pages deploy
```

## Deployment

`.github/workflows/pages.yml` builds `web/` on every push to `main` and deploys it with GitHub Actions Pages. The Vite `base` is `/BetterRankings/` for the project site. `.github/workflows/ci.yml` runs the Python tests, the TypeScript tests and the build on every push.

## License

MIT
