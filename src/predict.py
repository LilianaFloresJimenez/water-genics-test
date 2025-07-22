import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class ModelPrediction:
    def predict_and_compare_from_csv(self, csv_file_path: str, model_name: str) -> None:
        xs, y, times = self.load_values_from_csv(csv_file_path)
        self.predict_and_compare(xs, y, model_name, times)

    def predict_and_compare(self, xs, y, model_name, times):
        y_predictions = self._predict(xs, y, model_name)
        differences = y - y_predictions
        results_df = pd.DataFrame({
            'Timestamp': times,
            'Actual': y.flatten(),
            'Predicted': y_predictions.flatten(),
            'Difference': differences.flatten()
        })
        row_count = 5
        print("\nPrediction Results (first %s rows):" % row_count)
        print(results_df.head(row_count).to_string())
        print(f"\nTotal number of predictions: {len(y_predictions)}")

    def load_values_from_csv(self, csv_file_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_frame = pd.read_csv(csv_file_path)
        xs = data_frame.filter(regex='^input_')
        y = data_frame['target_variable'].values.reshape(-1, 1)
        times = data_frame['time']
        return xs, y, times

    def _predict(self, xs, y, model_name: str):
        mlflow.set_tracking_uri("file://" + os.path.abspath(os.path.dirname(__file__) + "/../mlruns"))
        model = mlflow.sklearn.load_model("models:/%s/latest" % model_name)
        y_predictions = model.predict(xs)
        metrics = {
            'r2_score': r2_score(y, y_predictions),
            'mean_absolute_error': mean_absolute_error(y, y_predictions),
            'mean_squared_error': mean_squared_error(y, y_predictions),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y, y_predictions))
        }

        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return y_predictions


def main():  # pragma: no cover
    model_prediction = ModelPrediction()
    model_prediction.predict_and_compare_from_csv(
        os.path.dirname(__file__) + "/../data/202302_data_test.csv",
        "dynamic_example_1_model"
    )
    model_prediction.predict_and_compare_from_csv(
        os.path.dirname(__file__) + "/../data/202502_data_train.csv",
        "dynamic_example_2_model"
    )

    xs, y, times = model_prediction.load_values_from_csv(
        os.path.dirname(__file__) + "/../data/202302_data_test.csv"
    )
    train_xs, test_xs, train_y, test_y = train_test_split(xs, y, test_size=0.2, shuffle=False)
    test_times = times.tail(len(test_xs))
    model_prediction.predict_and_compare(test_xs, test_y, "dynamic_example_3_model", test_times)


if __name__ == "__main__":  # pragma: no cover
    main()
