# Industrial Predictive Maintenance

**Work in progress**

An end-to-end machine learning project for monitoring the health of rotating machinery, built on real vibration data from bearings that were run until failure.

## Goal

Detect early signs of bearing wear and estimate the remaining useful life of a bearing, so maintenance can be planned before a breakdown happens.

## Planned work

- Process real run-to-failure vibration data
- Train and evaluate models for failure risk and remaining useful life
- Build an MLOps pipeline with automated testing, CI/CD, Docker, model versioning, monitoring and retraining
- Serve predictions through an API and a live dashboard

## Status

The project environment is set up. The parts above are not built yet. This README is updated as each part is finished.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv installs it automatically)

## Setup

```bash
git clone https://github.com/Hossain007Bhuiyan/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
uv sync
```
