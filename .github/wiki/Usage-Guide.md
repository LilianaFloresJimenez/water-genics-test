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