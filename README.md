# StadioEquities Predicting Account Activation

1. Motivation

STADIOEquities business model depends on converting registered accounts into funded, active investors. An empty account earns the platform almost nothing, so activation is not a soft metric it is, in the company's own words, the entire economic model. But form the platform's 2.3 million registered accounts, 41% have never deposited a single rand and the sign up to first deposit conversion rate has dropped from 64% to 59% over the last two years. Over the same period accounts becoming dormant within six months have risen from 22% to 31% and KYC-step abandonment during onboarding has climbed from 13% to 18%.

The cost of this gap is direct, acquiring an account costs the business R180 a cost that is "recovered only if the account activates." Every account in the 41% that never funds, represents acquisition spend the platform cannot recoup while marketing continues to add new registrations faster than the platform activates the ones it already has. Compounding the problem, onboarding communications currently go out "on a fixed schedule to everyone, blind to who is about to activate and who is about to disappear," even though the briefing pack notes the drop-off "is not random", it clusters by acquisition channel, onboarding progress, first-session behaviour, and time-to-first-deposit.

STADIOEquities already captures the behavioural detail needed to address this. "Every screen viewed, every deposit, every trade, every abandoned sign-up step." 

A model that uses this existing data to flag, early in the registration journey, which accounts are trending toward activation and which are trending toward abandonment would let the growth and onboarding teams move from blanket nudging to targeted, individualised intervention. This directly supports the first pillar of the STADIOEquities 2030 strategy — to "activate the accounts we already have"  and puts a firmer floor under a revenue base (R1.4bn, up 22% year on year) that depends almost entirely on clients who fund and keep using their accounts.

2. Problem Statement

STADIOEquities cannot currently distinguish at the point of registration or during onboarding, which newly registered accounts are likely to fund and become active from which are likely to abandon the sign-up process or drift into dormancy. In the absence of this distinction, every registrant receives the same fixed-schedule onboarding communications regardless of their individual trajectory, and the R180 spent acquiring each account is left to convert — or not — without any targeted support.

This project addresses the following problem:

Can a predictive model, built on STADIOEquities' app and web behavioural data, onboarding progress, and account and funding history, identify early in the registration and onboarding journey, which newly registered accounts are at high risk of never funding or of stalling during KYC, so that the platform can intervene with targeted and individualised nudges instead of a blanket communication schedule?

Framed as a data science task, this is a binary classification problem: given behavioural, funding, and demographic features captured up to a defined early checkpoint (e.g. day 7 or day 14 post-registration), predict whether an account will go on to fund and remain active within a defined outcome window (e.g. 90 days). Historical account-level outcomes (funded/never-funded, active/dormant) provide the labels needed to train and validate the model using standard supervised learning techniques.

3. Repository Structure

```
.
├── README.md                           # Project overview and repository guide
├── Data/
│   ├── raw/                            # Unmodified data received from STADIOEquities
│   └── processed/                      # Cleaned and feature-engineered datasets
├── data_request/
│   ├── STADIOEquities_Data_Request.pdf # Formal data request (Part C)
│   └── configuration files/             # Data request configuration material
├── experiments/                        # Experimental notebooks and configuration files
├── findings/
│   └── scripts/                        # Scripts supporting analysis and findings
├── models/                              # Saved trained models and model artefacts
├── results/                             # Metrics, plots, and experiment reports
└── requirements.txt                     # Python dependencies
```

The project directories are working areas. Files will be added to their relevant directories as the project progresses.

Where each required artefact lives:

| Artefact | Location |
| --- | --- |
| Datasets | `Data/raw/`, `Data/processed/` |
| Models | `models/` |
| Experimental setup | `experiments/` |
| Findings and analysis scripts | `findings/scripts/` |
| Experimental results | `results/` |
| Data request (Part C) | `data_request/` |

4. RAAIDD Log

| RAAIDD Element | Project-Specific Entries |
| --- | --- |
| **Risks** | 1. Behavioural logging may not extend far enough back to cover first-session activity for accounts registered before tracking began, shrinking the usable training sample for the earliest cohorts.<br>2. Class imbalance may cause a naively trained model to predict that most accounts will not activate and miss the accounts most worth targeting.<br>3. Behavioural events from the app and web platforms may not join cleanly to a single account ID, leaving gaps in onboarding-step histories.<br>4. Changes to the onboarding flow or KYC steps may reduce the transferability of historically derived features at deployment. |
| **Actions** | 1. Profile and document the account-ID join across the Account & Registration, Behavioural, Demographics, and Marketing extracts before feature engineering starts.<br>2. Reconstruct the "funded within 90 days" outcome label and compare it with the platform's reported 59% conversion rate.<br>3. Compare class-imbalance techniques, such as class weighting and SMOTE, and document the results in `results/` before selecting the final modelling approach.<br>4. Hold a checkpoint review with the STADIOEquities analytics stakeholder once early feature-importance results are available. |
| **Assumptions** | 1. Account IDs are consistent and joinable across all four requested data sources once pseudonymised.<br>2. The onboarding flow, KYC steps, and nudge schedule will remain materially unchanged during the project.<br>3. Activation is operationalised as a first deposit within 90 days of registration, consistent with the platform's "funded & active" definition. |
| **Issues** | 1. Historical KYC-completion timestamps may be missing or incomplete for some older accounts, delaying calculation of onboarding-duration features.<br>2. The reasons some historical accounts were closed may not be documented, making those accounts unsafe to label until clarified with the client. |
| **Decisions** | 1. Use a 90-day post-registration outcome window to align with STADIOEquities' definition of a "funded & active" account.<br>2. Treat "never funded" and "funded, then dormant within 6 months" as separate outcome classes because activation and retention have different drivers. |
| **Dependencies** | 1. Feature engineering depends on confirming the account-ID join across all four data sources.<br>2. The class-imbalance trial depends on validating the outcome label against the platform's reported conversion rate.<br>3. Stakeholder sign-off on the modelling approach depends on the feature-importance checkpoint review. |