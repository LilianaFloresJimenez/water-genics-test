import os

import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class FixedValuesModelTrainer:
    def get_fixed_data_points(self) -> tuple[np.array,np.array]:
        """Return some static example data for training linear regression model."""
        xs = np.array([
            [0.032707, 981.279743],
            [0.029623, 980.90025],
            [0.04886, 981.098875],
            [0.04912, 980.966079],
            [0.072415, 980.819118]
        ])
        y = np.array([
            33.806626,
            33.806626,
            67.613252,
            67.613252,
            84.516565
        ])
        return xs, y

    def train(self, xs: np.array, y: np.array, name: str) -> None:
        """Train linear regression model."""
        model_name = name + "_model"
        experiment_name = name + "_experiment"

        self._setup_mlflow(experiment_name)
        with mlflow.start_run():
            model = LinearRegression()
            model.fit(xs, y)
            y_predictions = model.predict(xs)
            metrics = {
                'r2_score': r2_score(y, y_predictions),
                'mean_absolute_error': mean_absolute_error(y, y_predictions),
                'mean_squared_error': mean_squared_error(y, y_predictions),
            }

            sample_count = len(xs)
            mlflow.log_params({
                'sample_count': sample_count,
                'intercept': model.intercept_.tolist(),
                'coefficients': model.coef_.tolist(),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                input_example=xs,
                registered_model_name=model_name,
            )

            print("\nMetrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    def _setup_mlflow(self, experiment_name):
        mlflow.set_tracking_uri("file://" + os.path.abspath(os.path.dirname(__file__) + "/../mlruns"))
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

def main(): # pragma: no cover
    xs, y = FixedValuesModelTrainer().get_fixed_data_points()
    FixedValuesModelTrainer().train(xs, y, 'fixed')

if __name__ == "__main__": # pragma: no cover
    main()