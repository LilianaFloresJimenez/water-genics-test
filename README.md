# water-genics application test

---

# Docs

## Installation Guide
Make sure virtualenv is installed on your system and pip is up to date.
It's possible to install the project with [just](https://github.com/casey/just):
```shell
    just setup
```
Alternatively, you can install it manually:
```shell
    # Installs python virtual environment
    virtualenv .venv
    . .venv/bin/activate
    # Installs dependencies from requirements.txt
    pip --require-virtualenv install -r requirements.txt
    # Installs this project (package) in developer mode
    pip --require-virtualenv install -e .
```

---

## Usage Guide
### Examples
The tasks are implemented in `src/train_fixed.py`, `src/train_dynamic.py`, and `src/predict.py`. Their `main` functions execute some examples.

You can run the examples like this:
```shell
    python src/train_fixed.py
    python src/train_dynamic.py
    python src/predict.py
```
Note that the examples in `predict.py` depend on the successful execution of the `train_dynamic.py` examples.

### MLflow
`src/train_fixed.py` and `src/train_dynamic.py` log data and models to MLflow. `predict.py` fetches a trained model from MLflow.

The MLflow UI can be started using:
```shell
    just mlflow-show-ui
    # Reloading your browser without cache may sometimes be necessary.
```
Or manually:
```shell
    mlflow ui
    # ... and then opening http://127.0.0.1:5000 in your browser
```

### Tests
Tests can be executed using:
```shell
    just test
```
Or manually:
```shell
    pytest tests/ -v --cov=src
```

---

## Troubleshooting:
- In my case I had issues to setup a virtual env. The issue was that my python was a pyenv python. I needed to make sure that pyenv is disabled for this project.
- Sometimes the mflow ui shows "Forbidden" messages. Then a browser reload without cache helps.
- The mocking in tests seems to work unreliable across different operation systems. On Mac Os it helped to mock also
`patch('mlflow.sklearn') as mock_sklearn` 

---

## AI Usage Report
I used the following tool:
- Google Gemini AI Chat

I consulted AI on the following questions and topics:
- Resolve project setup issues python interpreter / virtualenv.
- Introduction to MLflow and how to use it.
- How to mock dependencies in Python or pytest.
- How to generate an example JSON file from given example CSV files for tests.

---

# Tasks
- [x] Set up Python / Python virtual environment
- [x] Add suggested dependencies: numpy, pandas, pytest, scikit-learn and mlflow
- [x] Learn about mlflow and experiment with it
- [x] Create first version of task 1: "linear regression with static input"
- [x] Create first version of task 2: "linear regression with dynamic input"
- [x] Create first version of task 3: "Prediction with pretrained model"
- [x] Add tests for the three scripts, refactor, and extend to complete all tasks
- [x] Test code with examples (see `main` functions)
- [x] Add test code coverage dependencies (pytest-cov and coverage) and improve tests