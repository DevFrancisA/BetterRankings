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
- **Overlap:** the random logging policy gives good overlap. The effective sample size is 27–35% of rows, and the largest weight is about 8.

**2. The random policy from BTS's logs.**

- **What is estimated:** the uniform random policy's click rate, from BTS's logs alone.
- **Why it's clean:** every quantity is exact. The logged propensities are BTS's own, and the target policy is uniform.
- **Overlap:** poor. BTS rarely showed some items, so single weights reach 12,500 to 29,412. The effective sample size falls to 0.04–0.16% of rows, and the poor-overlap warning fires in every campaign.

## Method

- **Estimand:** clicks per impression slot, estimated position by position. A row's weight is `target(item | position) / logging(item | position)`.
- **Reward model:** a ridge logistic regression with these features:
  - item, position, day and hour of day;
  - the 4 categorical user features;
  - the user's affinity for the shown item.

  It is cross-fitted over 2 folds of recommendations, and each fold's model is trained on up to 1M sampled rows.
- **Weight clipping:** none. Clipping at 20 biased the random-from-BTS estimates.
- **Intervals:** 95% percentile bootstrap with 200 resamples. Resampling is done over recommendations, not rows: rows logged within 20 ms of each other are grouped, which puts 94% of rows in groups of exactly 3. The truth's standard error is clustered the same way.

## Results

"Error" is `|estimate − truth| / truth`. "✓" means the 95% interval contains the true click rate.

### BTS from the random policy's logs (weekly-average target)

| Campaign | True BTS click rate | Logged click rate, no correction | IPS | Self-normalised IPS | Reward model | Doubly robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.00495 | 30.0% off | 1.7% ✓ | 1.6% ✓ | 0.6% | 1.5% ✓ |
| men | 0.00674 | 24.0% off | 8.3% | 6.9% ✓ | 6.5% | 7.2% |
| women | 0.00638 | 24.5% off | 0.6% ✓ | 0.8% ✓ | 0.1% ✓ | 0.2% ✓ |

**Lift.** BTS's real click-rate lift over the random policy was 42.8% (all), 31.6% (men) and 32.5% (women). Doubly robust estimated it from the random policy's logs alone as 45.0%, 22.1% and 32.2%.

Under uniform logging, the naive "matched replay" estimator is identical to self-normalised IPS. So the table's naive baseline is instead the logged click rate with no correction: assume the new policy will do as well as the old one.

### The random policy from BTS's logs

| Campaign | True random-policy click rate | Logged click rate, no correction | IPS | Self-normalised IPS | Reward model | Doubly robust | Largest weight | Effective sample size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.00347 | 42.8% off | 1.0% ✓ | 5.0% | 5.1% | 3.4% ✓ | 12,500 | 19,910 (0.16%) |
| men | 0.00512 | 31.6% off | 2.5% ✓ | 7.0% | 5.0% | 6.9% ✓ | 29,412 | 1,513 (0.04%) |
| women | 0.00482 | 32.5% off | 4.4% | 8.0% | 1.8% | 7.1% | 21,739 | 6,009 (0.08%) |

### Sensitivity to how the BTS target is defined

Doubly robust error under three definitions of BTS's choice probabilities:

| Campaign | Weekly average (main) | Hourly behaviour | Published production prior |
| --- | ---: | ---: | ---: |
| all | 1.5% | 5.4% | 5.1% |
| men | 7.2% | 0.1% | 8.8% |
| women | 0.2% | 10.4% | 0.3% |

## What this shows, and what it doesn't

- **The weighting estimators beat the uncorrected comparison.** Assuming a new policy will match the old one's click rate misses by 24–43%. Across all 6 experiments IPS has a median error of 2.1%, and 5 of the 6 land within 0.6–4.4%; the exception is men, at 8.3%.
- **Precision is limited by the data, not just the estimators.** At click rates near 0.5%, the intervals for BTS-from-random span roughly ±5–9%. Most differences between estimators in a single campaign are within that noise.
- **The target definition matters as much as the estimator.** The hourly target looked like a fix for the men campaign, but it made the other two worse, so it isn't used for the headline numbers. BTS's published prior also differs from what BTS actually did: the item mix at each position differs by 31–52% (total variation distance). That gap is part of any benchmark built on the prior.
- **Reward-model intervals are too narrow.** The bootstrap does not refit the model, so its intervals exclude fitting uncertainty, and they miss the truth even when the point estimate is close.
- **Assumptions the position-wise estimators make:**
  - A click at one position doesn't depend on which items fill the other two positions. Random and BTS slates differ in this respect, which is a likely source of the remaining error.
  - BTS's weekly-average behaviour stands in for a policy that changed during the week.

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
