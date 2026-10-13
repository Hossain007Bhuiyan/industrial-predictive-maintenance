# Industrial Predictive Maintenance

**Work in progress**

An end-to-end machine learning project for monitoring the health of rotating machinery, built on real vibration data from laboratory bearing tests and an industrial pulp mill.

## Goal

Detect early signs of bearing wear and estimate the remaining useful life of a bearing, so maintenance can be planned before a breakdown happens.

## Planned work

- Process real vibration data from bearings
- Train and evaluate models for failure risk, remaining useful life and fault detection
- Build an MLOps pipeline with automated testing, CI/CD, Docker, model versioning, monitoring and retraining
- Serve predictions through an API and a live dashboard

## Status

Done:

- Data download with SHA-256 checksum verification
- Raw data audit, loaders and tests for all three datasets
- Data overview and exploration notebooks
- MQTT broker (Mosquitto in Docker) and a replay service that streams real recordings as if the sensors were live
- Drift injector for testing drift detection later
- Feature extraction: time domain, frequency bands and envelope spectrum amplitudes at the bearing fault frequencies
- Ingestion service that validates every MQTT message against a schema, computes its features and rejects broken messages with a reason
- Feature tables for all three datasets, checked with a Pandera schema, using the same feature code as the live stream
- Features notebook

The other parts are not built yet. This README is updated as each part is finished.

## Data

| Dataset | Source | License |
| --- | --- | --- |
| FEMTO bearing dataset | PRONOSTIA test rig, FEMTO-ST Institute (IEEE PHM 2012 challenge), via the NASA Prognostics Data Repository | Not stated by the provider |
| IMS bearing dataset | NSF I/UCR Center for Intelligent Maintenance Systems, via data.nasa.gov | U.S. Government Works, as listed on data.nasa.gov |
| SCA bearing dataset | Mid Sweden University and SCA, pulp mill measurements from 2019 to 2022 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

No raw data is stored in this repository. The download script fetches FEMTO and IMS from their sources and checks every file against a SHA-256 checksum. SCA has to be downloaded by hand from [Mendeley Data](https://data.mendeley.com/datasets/tdn96mkkpt/2). The project only reads the raw files. Everything derived from them is produced by the code in this repository.

### Known data issues

Found while auditing and exploring the data:

- **FEMTO and IMS are laboratory tests with accelerated wear**, not measurements from a factory.
- **FEMTO end of life:** the literature describes a 20 g stop criterion. Four bearings never reach 20 g and others pass it long before the end. End of life is therefore taken as the last recording of each bearing.
- **FEMTO sensor limit:** the horizontal channel never goes above 48.148 g. Bearing1_1, 1_3, 1_4 and 2_3 have 32 recordings at that ceiling, so their highest peaks are cut off. The rig's Dytran 3035B accelerometers exist in ±50, ±100 and ±500 g versions. The version used is not documented, so the limit is taken from the data.
- **FEMTO clock values:** two files in Bearing1_1 (acc_02121 and acc_02122) carry a wrong time stamp. Their times are interpolated and marked in the data.
- **FEMTO format:** Bearing1_4 uses semicolons instead of commas. Bearing1_3, 2_2, 2_3 and 3_2 have no temperature files.
- **FEMTO low recordings:** Bearing2_1 recording 13 and Bearing3_2 recording 1562 have about ten times less vibration than the recordings around them. They are kept in the raw data and are to be left out when training models.
- **FEMTO wear pattern:** in most bearings the vibration level stays flat for most of the life and only rises near the end. No single feature follows the wear in all 17 bearings.
- **IMS test 3:** the archive holds 1,875 recordings after the documented end on 4 April 2004. Bearing 3 only degrades in that part. The full archive is used.
- **IMS stopped recordings:** two recordings at the end of test 2 and one at the end of test 3 contain no vibration and are left out.
- **IMS unit:** the vibration unit is not documented.
- **SCA labels** were set by hand from envelope spectra, so fault start and end dates can be slightly off. In case 10 the outer ring fault label starts about six weeks before the fault is visible at its fault frequency. Case 11 contains an external event that is labelled normal and is only used to test false alarms. Case 9 mixes two sampling rates.
- **SCA case 9 normal data:** in the normal period after the bearing change, the outer ring fault feature rises again from May 2021. That period must not be used as clean normal data when training models.

### Synthetic changes

The drift injector changes replayed real recordings on purpose to test drift detection. It never creates new data. Every changed message is marked as injected.

**References**

- Nectoux, P. et al. (2012). PRONOSTIA: An experimental platform for bearings accelerated degradation tests. IEEE International Conference on Prognostics and Health Management, Denver.
- Qiu, H., Lee, J. & Lin, J. (2006). Wavelet filter-based weak signature detection method and its application on roller bearing prognostics. Journal of Sound and Vibration, 289, 1066-1090.
- Lundström, A. & O'Nils, M. (2024). SCA bearing dataset. Mendeley Data, V2. https://doi.org/10.17632/tdn96mkkpt.2
- Lundström, A. & O'Nils, M. (2023). Factory-Based Vibration Data for Bearing-Fault Detection. Data, 8(7), 115. https://doi.org/10.3390/data8070115

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv installs it automatically)
- unar, to unpack the IMS archives (macOS: `brew install unar`, Ubuntu: `sudo apt install unar`)
- Docker with Docker Compose, for the MQTT broker
- About 14 GB of free disk space for the downloaded and unpacked data

## Setup

```bash
git clone https://github.com/Hossain007Bhuiyan/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
uv sync
```

Download the SCA dataset by hand from Mendeley Data with "Download All" and save it as `data/raw/sca_bearing.zip`. Then download and verify everything:

```bash
uv run python -m industrial_predictive_maintenance.data_download
```

Build the feature tables (about three minutes):

```bash
uv run python -m industrial_predictive_maintenance.build_features
```

Run the tests:

```bash
uv run pytest
```

## Live replay

Start the MQTT broker and replay one FEMTO bearing at ten times the recorded speed:

```bash
docker compose up -d mqtt
uv run python -m industrial_predictive_maintenance.replay femto Bearing1_1 --speed 10
```

Messages are published on topics like `ipm/femto/Bearing1_1/vibration`. In a second terminal, the ingestion service receives them, validates them and stores their features in `data/stream/`:

```bash
uv run python -m industrial_predictive_maintenance.ingest
```

Stop the broker with `docker compose down`.