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

The other parts are not built yet. This README is updated as each part is finished.

## Data

| Dataset | Source | License |
| --- | --- | --- |
| FEMTO bearing dataset | PRONOSTIA test rig, FEMTO-ST Institute (IEEE PHM 2012 challenge), via the NASA Prognostics Data Repository | Not stated by the provider |
| IMS bearing dataset | NSF I/UCR Center for Intelligent Maintenance Systems, via data.nasa.gov | U.S. Government Works, as listed on data.nasa.gov |
| SCA bearing dataset | Mid Sweden University and SCA, pulp mill measurements from 2019 to 2022 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

No raw data is stored in this repository. The download script fetches FEMTO and IMS from their sources and checks every file against a SHA-256 checksum. SCA has to be downloaded by hand from [Mendeley Data](https://data.mendeley.com/datasets/tdn96mkkpt/2). The project only reads the raw files. Everything derived from them is produced by the code in this repository.

### Known data issues

Found during the data audit and handled in the code:

- **FEMTO and IMS are laboratory tests with accelerated wear**, not measurements from a factory.
- **FEMTO end of life:** the literature describes a 20 g stop criterion. Four bearings never reach 20 g and others pass it long before the end. End of life is therefore taken as the last recording of each bearing.
- **FEMTO sensor limit:** Bearing1_1, 1_3, 1_4 and 2_3 reach exactly 48.15 g, so their highest peaks are cut off by the sensor range.
- **FEMTO clock values:** two files in Bearing1_1 (acc_02121 and acc_02122) carry a wrong time stamp. Their times are interpolated and marked in the data.
- **FEMTO format:** Bearing1_4 uses semicolons instead of commas. Bearing1_3, 2_2, 2_3 and 3_2 have no temperature files.
- **IMS test 3:** the archive holds 1,875 recordings after the documented end on 4 April 2004. Bearing 3 only degrades in that part. The full archive is used.
- **IMS stopped recordings:** two recordings at the end of test 2 and one at the end of test 3 contain no vibration and are left out.
- **IMS unit:** the vibration unit is not documented.
- **SCA labels** were set by hand from envelope spectra, so fault start and end dates can be slightly off. Case 11 contains an external event that is labelled normal and is only used to test false alarms. Case 9 mixes two sampling rates.

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

Messages are published on topics like `ipm/femto/Bearing1_1/vibration`. Stop the broker with `docker compose down`.