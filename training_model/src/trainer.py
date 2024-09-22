from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel, Pipeline
from pyspark.sql import Row
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import mlflow


class Trainer:
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize the Trainer class with paths to save the model and S3 bucket.

        :param model_path: Local or S3 path for saving the trained model
        """
        self.model_path = model_path
        self.data_path = data_path
        self.spark = self._initialize_spark()
        self.data = self.load_data()
        self.mlflow_tracking_uri = "http://158.160.24.72:8000"
        self.mlflow_experiment = "Speech detection"
        self.params: dict = {
            "num_features": 20000,
            "num_trees": 100,
            "max_depth": 15
        }

    @staticmethod
    def _initialize_spark() -> SparkSession:
        """
        Initialize a Spark session with pre-configured settings.

        :return: SparkSession object
        """
        spark = SparkSession.builder \
            .appName("Text Classification with PySpark") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.executor.instances", "2") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.storage.memoryFraction", "0.6") \
            .config("spark.memory.fraction", "0.8") \
            .getOrCreate()
        return spark

    def load_data(self) -> DataFrame:
        """
        Load data from a CSV file into a Spark DataFrame with 'Content' and 'Label' columns.

        :return: A Spark DataFrame containing the data from the CSV file, with null values dropped.
        """
        return self.spark.read.csv(self.data_path, header=True, inferSchema=True).na.drop()

    def train_model(self) -> PipelineModel:
        """
        Load data, preprocess it, and train the model.

        :return: Trained PipelineModel
        """
        train_data, _ = self.data.randomSplit([0.8, 0.2], seed=42)

        tokenizer = Tokenizer(inputCol="Content", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=self.params["num_features"]
        )
        idf = IDF(inputCol="raw_features", outputCol="features")
        indexer = StringIndexer(inputCol="Label", outputCol="indexedLabel")
        rf = RandomForestClassifier(
            labelCol="indexedLabel",
            featuresCol="features",
            numTrees=self.params["num_trees"],
            maxDepth=self.params["max_depth"]
        )

        pipeline = Pipeline(stages=[indexer, tokenizer, remover, hashing_tf, idf, rf])
        model = pipeline.fit(train_data)
        return model

    def evaluate_model(self, model: PipelineModel) -> dict:
        """
        Evaluate the model on the test data.

        :param model: Trained PipelineModel
        :return: Dictionary of evaluation metrics
        """
        _, test_data = self.data.randomSplit([0.8, 0.2], seed=42)

        predictions = model.transform(test_data)
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                         metricName="f1")
        precision_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                                metricName="weightedPrecision")
        recall_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                             metricName="weightedRecall")
        auc_evaluator = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction",
                                                      metricName="areaUnderROC")

        f1_score = round(f1_evaluator.evaluate(predictions), 3)
        precision = round(precision_evaluator.evaluate(predictions), 3)
        recall = round(recall_evaluator.evaluate(predictions), 3)
        auc = round(auc_evaluator.evaluate(predictions), 3)

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc", auc)
            for key, value in self.params.items():
                mlflow.log_param(key, value)

        return {
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

    def save_model(self, model: PipelineModel):
        """
        Save the trained model locally.

        :param model: Trained PipelineModel
        """
        model.write().overwrite().save(self.model_path)
        print(f"Model uploaded to S3 bucket: {self.model_path}")

    def predict(self, new_comment: str) -> dict:
        """
        Load a model from the given path and predict the label for a new comment.

        :param new_comment: New comment text for prediction
        :return: Dictionary with predicted label and probabilities
        """
        model = PipelineModel.load(self.model_path)
        data = [Row(Content=new_comment)]
        new_data_df = self.spark.createDataFrame(data)

        predictions = model.transform(new_data_df)
        predicted_label = predictions.select("prediction").collect()[0]["prediction"]
        probability = predictions.select("probability").collect()[0]["probability"]

        return {
            "comment": new_comment,
            "predicted_label": int(predicted_label),
            "probability": probability
        }


if __name__ == '__main__':
    trainer = Trainer(
        model_path="s3a://amamylov-mlops/hate_speech_detection/model",
        data_path="s3a://amamylov-mlops/hate_speech_detection/train_dataset_1.csv"
    )
    model = trainer.train_model()
    metrics = trainer.evaluate_model(model)

    trainer.save_model(model)
    prediction = trainer.predict(
        new_comment="This is a good comment!"
    )
    print(metrics)
    print(prediction)