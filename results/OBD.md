# Real-data check: Open Bandit Dataset

These are measured results from running BetterRankings' estimators on real logs, produced by `betterrankings obd --campaign every --prior prior_bts.yaml --json obd_results.json`. The raw output is in [`obd_results.json`](obd_results.json).

## The data

The [Open Bandit Dataset](https://research.zozo.com/data.html) (CC BY 4.0) comes from ZOZOTOWN, a large fashion e-commerce site in Japan.

For one week in November 2019, two recommendation policies ran side by side on live traffic:

- a **uniform random** policy;
- **Bernoulli Thompson sampling (BTS)**.

Each recommendation shows 3 items (left, center and right) from 34 to 80 candidates. Every log row records:

- the item shown at one position;
- whether it was clicked;
- the logging policy's probability of that choice.

The data comes in three campaigns: all, men and women.

| Campaign | Items | Random-policy rows | BTS rows |
| --- | ---: | ---: | ---: |
| all | 80 | 1,374,327 | 12,357,200 |
| men | 34 | 452,949 | 4,077,727 |
| women | 46 | 864,585 | 7,765,497 |

Both policies were logged on the same traffic, so each one's real click rate is known. That turns off-policy evaluation into a test with a real answer: estimate one policy's click rate from the other policy's logs, then compare.

## Two experiments

**1. BTS from the random policy's logs.**

- **What is estimated:** BTS's click rate per impression slot, from the random logs alone. The truth is BTS's on-policy click rate.
- **Target policy:** the importance weights need BTS's probability of showing each item at each position. Its logged propensities flip between a few values within minutes, so its per-impression policy can't be rebuilt exactly. The main results use BTS's **weekly average behaviour**: how often each item appeared at each position across BTS's own log.
- **Context:** BTS's choices barely depend on the user features. The total variation between user segments is 0.02 to 0.09, so treating it as non-contextual is reasonable.
- **Overlap:** the random logging policy gives good overlap.

**2. The random policy from BTS's logs.**

- **What is estimated:** the uniform random policy's click rate, from BTS's logs alone.
- **Target policy:** uniform, so it is known exactly.
- **Overlap:** poor. BTS rarely showed some items, so their weights are heavy.

## Method

- **Estimand:** clicks per impression slot, estimated position by position. A row's weight is `target(item | position) / logging(item | position)`.
- **Logging propensities:** each experiment runs twice.
  - **Logged:** the propensities recorded in the data.
  - **Estimated (daily):** re-estimated from the log itself, as how often each item appeared at each position on each day. Weighting with estimated propensities is never less efficient than with the true ones (Hirano, Imbens and Ridder, 2003). It also removes the miscalibration found in BTS's logged values (see below).

  The daily bucket was chosen before seeing any results: it is the finest time bucket whose item-position cells are all filled.
- **Reward model:** a ridge logistic regression with these features:
  - item, position, day and hour of day;
  - the 4 categorical user features;
  - the user's affinity for the shown item.

  It is cross-fitted over 2 folds of recommendations, and each fold's model is trained on up to 1M sampled rows.
- **Weight clipping:** none.
- **Intervals:** 95% percentile bootstrap with 200 resamples. Resampling is done over recommendations, not rows: rows logged within 20 ms of each other are grouped, which puts 94% of rows in groups of exactly 3. The truth's standard error is clustered the same way.

## Results

Errors are signed: `(estimate − truth) / truth`. "±" is the half-width of the 95% interval, relative to the truth. "✓" means the interval contains the true click rate.

### The random policy from BTS's logs

This is where estimated propensities matter.

| Campaign | Propensities | IPS | Self-normalised IPS | Doubly robust | Effective sample size | Largest weight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | logged | −1.0% ±5.0% ✓ | −5.0% ±4.9% | −3.4% ±5.0% ✓ | 0.16% | 12,500 |
| all | estimated | −4.0% ±1.8% | −4.0% ±1.7% | −2.3% ±1.7% | 15.7% | 159 |
| men | logged | +2.5% ±4.9% ✓ | −7.0% ±6.1% | −6.9% ±7.0% ✓ | 0.04% | 29,412 |
| men | estimated | −4.0% ±2.8% | −4.0% ±2.8% | −1.5% ±2.8% ✓ | 19.3% | 69 |
| women | logged | −4.4% ±3.2% | −8.0% ±3.7% | −7.1% ±3.7% | 0.08% | 21,739 |
| women | estimated | −4.9% ±1.9% | −4.9% ±1.9% | −4.5% ±1.9% | 12.8% | 152 |

Re-estimating the propensities has these effects in all three campaigns:

- The effective sample size rises from 0.04–0.16% of rows to 13–19%.
- The largest weight falls from 12,500–29,412 to 69–159.
- Intervals become 2 to 3 times narrower.
- Doubly robust error falls: from 3.4% to 2.3%, from 6.9% to 1.5%, and from 7.1% to 4.5%.
- Self-normalised IPS error falls: from 5.0% to 4.0%, from 7.0% to 4.0%, and from 8.0% to 4.9%.

Plain IPS does not improve: it goes from −1.0%, +2.5% and −4.4% to −4.0%, −4.0% and −4.9%. Its logged-propensity estimates had wide intervals that happened to land near the truth. The tighter estimates expose a consistent bias of about −4%, explained below.

The uncorrected comparison, assuming the random policy would match BTS's logged click rate, is 31.6% to 42.8% off.

### BTS from the random policy's logs (weekly-average target)

| Campaign | Propensities | IPS | Self-normalised IPS | Reward model | Doubly robust |
| --- | --- | ---: | ---: | ---: | ---: |
| all | logged | +1.7% ±5.1% ✓ | +1.6% ±5.1% ✓ | +0.6% | +1.5% ±5.0% ✓ |
| all | estimated | +1.8% ±5.2% ✓ | +1.8% ±5.2% ✓ | +0.6% | +1.8% ±5.2% ✓ |
| men | logged | −8.3% ±6.7% | −6.9% ±6.8% ✓ | −6.5% | −7.2% ±6.7% |
| men | estimated | −6.1% ±7.2% ✓ | −6.1% ±7.2% ✓ | −6.5% | −6.6% ±7.4% ✓ |
| women | logged | +0.6% ±4.6% ✓ | +0.8% ±4.5% ✓ | −0.1% ✓ | −0.2% ±4.5% ✓ |
| women | estimated | +0.3% ±4.5% ✓ | +0.3% ±4.4% ✓ | −0.1% ✓ | −0.8% ±4.3% ✓ |

The random policy's logged propensities are exact (1 over the number of items), so re-estimating changes little here. With estimated propensities, every IPS, self-normalised and doubly robust interval covers the truth.

The uncorrected comparison, assuming BTS would match the random policy's click rate, is 24.0% to 30.0% off.

**Lift.** BTS's real click-rate lift over the random policy was 42.8% (all), 31.6% (men) and 32.5% (women). Doubly robust estimated it from the random policy's logs alone (logged propensities) as 45.0%, 22.1% and 32.2%.

### Sensitivity to how the BTS target is defined

Doubly robust error (logged propensities) under three definitions of BTS's choice probabilities:

| Campaign | Weekly average (main) | Hourly behaviour | Published production prior |
| --- | ---: | ---: | ---: |
| all | +1.5% | +5.4% | −5.1% |
| men | −7.2% | +0.1% | −8.8% |
| women | −0.2% | +10.4% | −0.3% |

## What limits accuracy: three findings

**1. BTS's logged propensities are miscalibrated.**

- Under BTS's logs, the uniform policy's weights should average exactly 1. They average 1.04 to 1.10, between 3 and 6 standard errors too high, and individual day-and-position cells reach 3.2.
- Rows logged with a propensity below 1e-4 have items that appear at that position about 3 times in 1,000 that day, 30 to 300 times more often than logged.
- This is consistent with the propensities having been computed by Monte Carlo simulation, which is unreliable for rare items.
- Re-estimating the propensities fixes it: the mean weight becomes exactly 1.000.

**2. Slate competition.**

The estimators assume a click at one position doesn't depend on the items shown at the other two. That assumption fails here.

- **The test:** the same item, at the same position, on the same day, was clicked less often in BTS's slates than in random slates.
- **The size:** the ratio of click rates was 0.957 (all), 0.965 (men) and 0.950 (women). The 95% cell-bootstrap intervals were 0.910–1.006, 0.896–1.044 and 0.905–0.998.
- **The cause:** BTS fills the other slots with strong items, which take clicks away.
- **The consequence:** estimates of the random policy from BTS's logs come out about 4–5% low. That matches the −4.0%, −4.0% and −4.9% of IPS with estimated propensities.

**3. Timing.**

- **The mechanism:** BTS learned during the week and showed items more often while they were clicking well. A fixed target or propensity, weekly or daily, misses that.
- **Its direction:** it pushes estimates the other way from slate competition, by a similar amount.
- **Finer time buckets don't fix it.** Hourly estimated propensities leave many hour cells empty, so part of the target mass silently drops out. Random-from-BTS IPS falls to −11.7% to −16.7% with hourly buckets.
- **A misleading coincidence:** ignoring time entirely (one propensity per item and position for the whole week) gives near-zero error (+0.1%, +2.0% and +0.5%). That is the timing bias and the slate bias cancelling, not a better method, so it is not used.

Removing the remaining few percent would need modelling both biases at once. The data can't cleanly separate them: they are of similar size, they pull in opposite directions, and the intervals for BTS-from-random are ±4–7%.

## Caveats

- **Reward-model intervals are too narrow.** The bootstrap does not refit the model, so its intervals exclude fitting uncertainty.
- **Three campaigns is a small sample.** Differences of a few percent between estimators in one campaign are mostly within the noise.
- **The target-policy definition matters as much as the estimator.** BTS's published prior differs from what BTS actually did: the item mix at each position differs by 31–52% (total variation distance). Any benchmark built on the prior inherits that gap.

## Reproduce

```bash
# about 413 MB download, 11 GB unzipped
curl -O https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip open_bandit_dataset.zip
curl -O https://raw.githubusercontent.com/st-tech/zr-obp/master/obp/policy/conf/prior_bts.yaml

betterrankings obd --data open_bandit_dataset --campaign every --prior prior_bts.yaml \
    --cache obd_cache --json obd_results.json
```

The first run parses the CSVs (a few minutes for the 6.3 GB BTS log). Later runs reuse the cache.

## Citation

Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita. "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation." NeurIPS Datasets and Benchmarks Track, 2021. Data © ZOZO Technologies, CC BY 4.0.

Keisuke Hirano, Guido W. Imbens, Geert Ridder. "Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score." Econometrica 71(4), 2003.
