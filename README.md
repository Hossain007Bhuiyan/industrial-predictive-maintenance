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

Done: data download with checksum verification, a raw data audit and loaders for all three datasets. The other parts are not built yet. This README is updated as each part is finished.

## Data

| Dataset | Source | License |
| --- | --- | --- |
| FEMTO bearing dataset | PRONOSTIA test rig, FEMTO-ST Institute (IEEE PHM 2012 challenge), via the NASA Prognostics Data Repository | Not stated by the provider |
| IMS bearing dataset | NSF I/UCR Center for Intelligent Maintenance Systems, via data.nasa.gov | U.S. Government Works, as listed on data.nasa.gov |
| SCA bearing dataset | Mid Sweden University and SCA, pulp mill measurements from 2019 to 2022 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

No raw data is stored in this repository. The download script fetches FEMTO and IMS from their sources and checks every file against a SHA-256 checksum. SCA has to be downloaded by hand from [Mendeley Data](https://data.mendeley.com/datasets/tdn96mkkpt/2). The project only reads the raw files. Everything derived from them is produced by the code in this repository.

**References**

- Nectoux, P. et al. (2012). PRONOSTIA: An experimental platform for bearings accelerated degradation tests. IEEE International Conference on Prognostics and Health Management, Denver.
- Qiu, H., Lee, J. & Lin, J. (2006). Wavelet filter-based weak signature detection method and its application on roller bearing prognostics. Journal of Sound and Vibration, 289, 1066-1090.
- Lundström, A. & O'Nils, M. (2024). SCA bearing dataset. Mendeley Data, V2. https://doi.org/10.17632/tdn96mkkpt.2
- Lundström, A. & O'Nils, M. (2023). Factory-Based Vibration Data for Bearing-Fault Detection. Data, 8(7), 115. https://doi.org/10.3390/data8070115

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv installs it automatically)
- unar, to unpack the IMS archives (macOS: `brew install unar`, Ubuntu: `sudo apt install unar`)

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