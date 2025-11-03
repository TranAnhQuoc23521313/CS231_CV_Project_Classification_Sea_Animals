# Computer Vision Project

Project scaffold for a computer vision pipeline.

Structure:
- configs/: YAML configs for data, model, training, evaluation, inference
- scripts/: helper scripts to prepare data, train, evaluate and run inference
- reports/: figures, experiment logs, reports
- data/: raw, processed, external, interim, samples
- notebooks/: exploratory and experiment notebooks
- src/: project source (data, models, training, evaluation, inference, utils)
- models/: checkpoints, logs, results

Getting started:
1. Create a Python environment (conda or venv).
2. Install dependencies from `requirements.txt` or `environment.yml`.
3. Place raw data in `data/raw/`.
4. Edit `configs/*.yaml` as needed.
5. Run `scripts/pipeline.sh` to run the end-to-end pipeline (stub).

