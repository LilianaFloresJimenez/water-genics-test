import os

import mlflow
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class DynamicValuesModelTrainer:

    def train_from_csv(self, csv_file_path, name: str):
        """Train model based on data from CSV file."""
        xs, y = self.load_values_from_csv(csv_file_path)
        self._validata_data(xs, y)
        self.train(xs, y, name)

    def train_from_json(self, json_file_path, name: str):
        """Train model based on data from JSON file."""
        df = pd.read_json(json_file_path)
        xs = pd.DataFrame(df['data'].apply(lambda x: x['input_variables']).tolist())
        y = pd.DataFrame(df['data'].apply(lambda x: x['target_variable']).tolist())
        self._validata_data(xs, y)
        self.train(xs, y, name)

    def train_from_lists(self, xs, y, name: str):
        """Train model based on data in simple python lists."""
        self._validata_data(xs, y)
        self.train(xs, y, name)

    def load_values_from_csv(self, file_path: str) -> tuple[DataFrame, DataFrame]:
        """Load data from CSV file."""
        data_frame = pd.read_csv(file_path)
        xs = data_frame.filter(regex='^input_')
        y = data_frame.filter(regex='^target_')
        return xs, y

    def _validata_data(self, xs: DataFrame, y: DataFrame) -> None:
        if len(xs) != len(y):
            raise ValueError("Input and target variables must have the same number of rows.")
        if len(xs) < 4:
            raise ValueError("Input and target variables must have at least 4 rows.")

    def train(self, xs, y, name: str) -> None:
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

    def _setup_mlflow(self, experiment_name: str) -> None:
        mlflow.set_tracking_uri("file://" + os.path.abspath(os.path.dirname(__file__) + "/../mlruns"))
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)


def main():  # pragma: no cover
    trainer = DynamicValuesModelTrainer()
    trainer.train_from_csv(os.path.dirname(__file__) + "/../data/202302_data_test.csv", 'dynamic_example_1')
    trainer.train_from_csv(os.path.dirname(__file__) + "/../data/202502_data_train.csv", 'dynamic_example_2')

    xs, y = trainer.load_values_from_csv(os.path.dirname(__file__) + "/../data/202302_data_test.csv")
    train_xs, test_xs, train_y, test_y = train_test_split(xs, y, test_size=0.2, shuffle=False)
    trainer.train_from_lists(train_xs, train_y, 'dynamic_example_3')


if __name__ == "__main__":  # pragma: no cover
    main()
