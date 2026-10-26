# SS3 — Dataset loading and preprocessing

**Notebook:** [`SS3_PartB.ipynb`](SS3_PartB.ipynb), Sections 1 and 2
**Input:** `Data/raw/client/*.csv` (four extracts, unmodified)
**Output:** `Data/processed/ss3_model_table.csv`

## 1. Loading

All four files are read as text (`dtype=str`) so nothing is silently converted before it has been checked.

| Extract | Rows | One row is | Used for |
|---|---|---|---|
| `1_account_registration.csv` | 5,433 (5,417 unique + 16 exact duplicates) | an account | registration date, channel, province, KYC timing, label |
| `2_app_web_behavioural.csv` | 42,644 events, 3,599 accounts | a screen / onboarding-step event | first-48-hour behaviour |
| `3_client_demographics.csv` | 5,417 | an account at sign-up | age, risk appetite, investment goal |
| `4_marketing_acquisition.csv` | 3,599 | an account | campaign, acquisition cost |

**Classes as delivered.** `ever_funded_flag`: 2,229 funded (41.1%) vs 3,188 never funded (58.9%), an imbalance of 1.43 : 1.

## 2. Data-quality problems found and how each was handled

| Problem | Evidence | Treatment |
|---|---|---|
| Duplicate accounts | 16 exact duplicate rows in table 1 | Dropped |
| Mixed date formats | 55 registration dates in `yyyy/MM/dd`, `dd/MM/yyyy` and `MM/dd/yyyy hh:mm` | Explicit parser. The day/month order of each shape was confirmed against the account's own KYC and first-event dates. A generic parser pushed one date past the extract date. |
| Inconsistent casing / whitespace | `' KZN '`, `kzn`, `gp` … | Stripped and upper-cased |
| Implausible values | Ages 104.62 and 110.07 | Set to missing, then imputed with the **training-set** median plus an `age_missing` flag |
| Blank categories | `stated_investment_goal`, `campaign_name` | Explicit "Not stated" / "No campaign" categories, kept because not answering can itself carry signal (same choice as SS2) |
| Duplicate column | `acquisition_channel` equals `registration_channel` for 100% of accounts | Kept one |
| README claims that don't hold | Mean acquisition cost is R148.97, not "close to R180"; the brief says 41% never deposited, but the extract shows 41% funded | Flagged for the client, not altered |

## 3. Fields used and fields excluded

| Excluded field | Why |
|---|---|
| `first_deposit_amount`, `current_balance`, `active_in_last_90_days_flag` | Post-outcome (leakage); dormancy is a separate problem |
| `kyc_status` | Snapshot taken at the 2026 extract, long after scoring time. Replaced by "KYC completed within 48 h" |
| `onboarding_nudge_sent_count`, `last_nudge_engagement` | Lifetime totals with no timestamp, so they can't be cut at the checkpoint |
| Behavioural events after the checkpoint | About a quarter of onboarding events, and 59% of step-7 "fund" events, happen after the first deposit |

## 4. Checkpoint, cohort and label

The model scores each account **48 hours after registration**, using only information available at that moment. This follows the SS2 rule that excluded call `duration`.

| Step | Accounts |
|---|---|
| Registered from Sep 2022 (behavioural + marketing history exists) | 3,599 |
| Full 90-day outcome window before the 2026-09-01 extract | 3,426 |
| Removed: already funded within 48 h (no nudge needed) | −563 |
| **Eligible accounts** | **2,863** |

**Label:** `activated = 1` if the first deposit arrives between 48 h and 90 days after registration.
**Classes:** 824 activated (28.8%) vs 2,039 not activated — imbalance 2.47 : 1.

## 5. Features (63 columns after one-hot encoding)

- **KYC:** completed within 48 h; hours to KYC (`-1` if not done)
- **Onboarding progress (≤ 48 h):** furthest step advanced and seen; steps abandoned; steps *returned to*
- **Engagement (≤ 48 h):** events, sessions, total session minutes, mean time on step, hours to first event, app share, used both app and web, distinct screens, deposit-screen and education-hub views, a "no activity in 48 h" flag
- **Static:** registration channel, province, campaign, acquisition cost (rand), registration hour and weekday, age, risk appetite, investment goal

Counts are zero when there were no events. Non-applicable timings use `-1` with a flag, the same sentinel-plus-flag pattern as `pdays` in SS2. Features are not scaled, because Random Forest splits don't depend on scale; the Logistic Regression benchmark scales inside its own pipeline.

## 6. Train / validation / test split

The split is **chronological by registration date**, as Part D recommended because of temporal drift:

| Subset | Accounts | Activated | Rate | Registered |
|---|---|---|---|---|
| Train (70%) | 2,004 | 598 | 29.8% | Sep 2022 – Apr 2025 |
| Validation (15%) | 429 | 127 | 29.6% | Apr 2025 – Nov 2025 |
| Test (15%, unseen) | 430 | 99 | 23.0% | Nov 2025 – Jun 2026 |

Validation is used for hyperparameter selection and the decision threshold. The test set is used once, for the final evaluation.
