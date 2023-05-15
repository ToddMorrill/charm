# CHARM
Repository for work done on the [DARPA CCU project](https://www.darpa.mil/news-events/2021-05-03a), where CHARM is the name of the Columbia University team and stands for Cross-Cultural Harmony through Affect and Response Mediation.

## Environment setup
You can install the necessary Python dependencies, including installing the `charm` Python library in this repository, (in a new virtual environment) by running
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m ipykernel install --user --name charm --display-name "Python (CHARM)"

# optionally launch Jupyter Lab
jupyter lab
```

## Layout of the repository
The core directories are as follows:
- `charm`: Python library for the project, with the following subdirectories:
    - `model`: subdirectory that contains all code for the social orientation and change point models. `train_pt.py` is the main entry point into all training code. See the module level docstring of that file for more details on how to train models.
    - `data`: contains code that prepared a metadata file for navigating the LDC data releases, which is now largely supplanted by the code in the `loaders` directory, which is a copy of the LDC data loader utilities written by Amith Ananthram and Ivan Dewerpe.
    - `eval`: contains a custom implementation of LDC's AP scoring function (`eval.py`) as well as some code used for interfacing with NIST's scoring tools.
- `circumplex`: contains code for generating the GPT annotated social orienation dataset. See that directory's `README.md` for more details.
- `integration`: contains Dockerfiles and other code for building our frontend and backend systems that interface with CCU queues and run our machine learning models, respectively. See `README.md` in that directory for more details.
- `demo`: contains code for the demo video that prepared for the March 2023 PI meeting.
- `miscellaneous`: contains various notebooks for ad-hoc analysis.
- `reports`: contains various reports and presentations that were prepared for the project.