import os
from textwrap import dedent
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.predict import ModelPrediction
from src.train_dynamic import DynamicValuesModelTrainer
from src.train_fixed import FixedValuesModelTrainer


def test_dependencies():
    import mlflow
    import sklearn
    import pandas
    import numpy
    import pytest
    import pytest_cov

    assert mlflow.__version__ >= "3.1"
    assert numpy.__version__ > "2.2"
    assert pandas.__version__ >= "2.3"
    assert pytest.__version__ >= "8.4"
    assert pytest_cov.__version__ >= "6.2"
    assert sklearn.__version__ >= "1.7"


def test_get_fixed_data_points():
    xs, y = FixedValuesModelTrainer().get_fixed_data_points()
    assert len(xs) == 5
    assert len(y) == 5


@pytest.fixture
def mock_mlflow():
    with (patch('mlflow.set_tracking_uri') as mock_set_tracking_uri, \
            patch('mlflow.get_experiment_by_name') as mock_get_experiment, \
            patch('mlflow.create_experiment') as mock_create_experiment, \
            patch('mlflow.set_experiment') as mock_set_experiment, \
            patch('mlflow.start_run') as mock_start_run, \
            patch('mlflow.log_params') as mock_log_params, \
            patch('mlflow.log_metrics') as mock_log_metrics, \
            patch('mlflow.sklearn.log_model') as mock_sklearn_log_model, \
            ):
        yield {
            'set_tracking_uri': mock_set_tracking_uri,
            'get_experiment': mock_get_experiment,
            'create_experiment': mock_create_experiment,
            'set_experiment': mock_set_experiment,
            'start_run': mock_start_run,
            'log_params': mock_log_params,
            'log_metrics': mock_log_metrics,
            'sklearn_log_model': mock_sklearn_log_model,
        }


def test_train_fixed_model(mock_mlflow: dict[str, MagicMock], capsys):
    xs, y = FixedValuesModelTrainer().get_fixed_data_points()
    FixedValuesModelTrainer().train(xs, y, 'fixed')
    output = capsys.readouterr().out
    assert output == dedent("""
        Metrics:
        r2_score: 0.9151
        mean_absolute_error: 5.5994
        mean_squared_error: 34.9155
        """)

    mock_mlflow['set_tracking_uri'].assert_called_once()
    mock_mlflow['get_experiment'].assert_called_once()
    mock_mlflow['create_experiment'].assert_not_called()
    mock_mlflow['set_experiment'].assert_called_once()
    mock_mlflow['start_run'].assert_called_once()
    mock_mlflow['log_params'].assert_called_once()
    mock_mlflow['log_metrics'].assert_called_once()
    mock_mlflow['sklearn_log_model'].assert_called_once()


@pytest.mark.parametrize("fixture_path,expected_count", [
    ("fixtures/4-columns.csv", 5),
    ("fixtures/5-columns.csv", 5)
])
def test_dynamic_trainer_load_input_variables_from_csv(fixture_path: str, expected_count: int):
    xs, y = DynamicValuesModelTrainer().load_values_from_csv(
        os.path.abspath(os.path.dirname(__file__) + "/" + fixture_path)
    )
    assert len(xs) == expected_count
    assert len(xs) == expected_count


def test_train_dynamic_model_from_csv(mock_mlflow: dict[str, MagicMock], capsys):
    DynamicValuesModelTrainer().train_from_csv(
        os.path.abspath(os.path.dirname(__file__) + "/fixtures/4-columns.csv"),
        'dynamic'
    )
    output = capsys.readouterr().out
    assert output == dedent("""
        Metrics:
        r2_score: 0.9151
        mean_absolute_error: 5.5994
        mean_squared_error: 34.9155
        """)

    mock_mlflow['start_run'].assert_called_once()


def test_train_dynamic_model_from_json(mock_mlflow: dict[str, MagicMock], capsys):
    DynamicValuesModelTrainer().train_from_json(
        os.path.abspath(os.path.dirname(__file__) + "/fixtures/test.json"),
        'dynamic'
    )
    output = capsys.readouterr().out
    assert output == dedent("""
        Metrics:
        r2_score: 0.9151
        mean_absolute_error: 5.5994
        mean_squared_error: 34.9155
        """)


def test_train_dynamic_model_from_lists(mock_mlflow: dict[str, MagicMock], capsys):
    DynamicValuesModelTrainer().train_from_lists(
        [
            [0.032707, 981.279743],
            [0.029623, 980.90025],
            [0.04886, 981.098875],
            [0.04912, 980.966079],
            [0.072415, 980.819118],
        ],
        [
            33.806626,
            33.806626,
            67.613252,
            67.613252,
            84.516565,
        ],
        'dynamic'
    )
    output = capsys.readouterr().out
    assert output == dedent("""
        Metrics:
        r2_score: 0.9151
        mean_absolute_error: 5.5994
        mean_squared_error: 34.9155
        """)


@pytest.mark.parametrize("xs,y,expected_error_message", [
    ([], [], "Input and target variables must have at least 4 rows."),
    (
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [1, 2, 3, 4, 5],
            "Input and target variables must have the same number of rows."
    ),
])
def test_train_dynamic_model_with_invalid_data(xs: list, y: list, expected_error_message: str):
    with pytest.raises(ValueError, match=expected_error_message):
        DynamicValuesModelTrainer().train_from_lists(xs, y, 'dynamic')


def test_predict(capsys):
    y_predictions = [
        45.715457,
        37.157540,
        55.861264,
        53.256488,
        54.742620,
    ]

    model = LinearRegression()
    model.predict = Mock(return_value=np.array(y_predictions).reshape(-1, 1))

    with patch('mlflow.sklearn.load_model', return_value=model):
        prediction = ModelPrediction()
        prediction.predict_and_compare_from_csv(
            os.path.abspath(os.path.dirname(__file__) + "/fixtures/5-columns.csv"),
            'dynamic_model'
        )
        captured = capsys.readouterr()
        output = captured.out
        assert output == dedent("""
            Metrics:
            r2_score: 0.9731
            mean_absolute_error: 0.9975
            mean_squared_error: 1.0464
            root_mean_squared_error: 1.0229
            
            Prediction Results (first 5 rows):
                               Timestamp  Actual  Predicted  Difference
            0  2023-02-15 07:00:00+00:00      47  45.715457    1.284543
            1  2023-02-15 10:54:00+00:00      38  37.157540    0.842460
            2  2023-02-15 14:51:00+00:00      55  55.861264   -0.861264
            3  2023-02-15 18:56:00+00:00      52  53.256488   -1.256488
            4  2023-02-15 22:53:00+00:00      54  54.742620   -0.742620
            
            Total number of predictions: 5
            """)
