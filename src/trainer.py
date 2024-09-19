import nltk
import mlflow
import mlflow.spark
from nltk.stem import WordNetLemmatizer
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import PipelineModel
from typing import Tuple, Dict


class Trainer:
    def __init__(self, data_path: str, num_features: int = 10000) -> None:
        """
        Initializes the Trainer class with a Spark session, data loading, and parameter setup.

        :param data_path: Path to the CSV file with data.
        :param num_features: Number of features for TF-IDF transformation.
        """
        self.spark = SparkSession.builder \
            .appName("Text Preprocessing with Lemmatization in PySpark") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.executor.instances", "2") \
            .getOrCreate()
        self.data_path = data_path
        self.num_features = num_features
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = Tokenizer(inputCol="Content", outputCol="words")
        self.remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        self.hashing_tf = HashingTF(inputCol="lemmatized_words", outputCol="raw_features", numFeatures=num_features)
        self.idf = IDF(inputCol="raw_features", outputCol="features")

    def lemmatize_words(self, words: list) -> list:
        """
        Lemmatizes a list of words.

        :param words: List of words to be lemmatized.
        :return: List of lemmatized words.
        """
        return [self.lemmatizer.lemmatize(word) for word in words]

    def preprocessing(self) -> Tuple[DataFrame, DataFrame]:
        """
        Preprocesses the data including tokenization, stop words removal, lemmatization, and TF-IDF transformation.

        :return: Tuple containing training and test datasets.
        """
        data = self.spark.read.csv(self.data_path, header=True, inferSchema=True)
        data = data.select("Content", "Label").na.drop()

        words_data = self.tokenizer.transform(data)
        filtered_data = self.remover.transform(words_data)

        lemmatize_udf = udf(lambda words: self.lemmatize_words(words), ArrayType(StringType()))
        lemmatized_data = filtered_data.withColumn("lemmatized_words", lemmatize_udf(col("filtered_words")))
        featurized_data = self.hashing_tf.transform(lemmatized_data)
        idf_model = self.idf.fit(featurized_data)
        rescaled_data = idf_model.transform(featurized_data)
        indexer = StringIndexer(inputCol="Label", outputCol="indexedLabel")
        final_data = indexer.fit(rescaled_data).transform(rescaled_data)
        train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

        return train_data, test_data

    def train_model(self, train_data: DataFrame) -> PipelineModel:
        """
        Trains a RandomForestClassifier model on the provided training dataset.

        :param train_data: Training dataset with features and labels.
        :return: Trained RandomForest model.
        """
        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=100, maxDepth=10)
        rf_model = rf.fit(train_data)
        mlflow.spark.log_model(rf_model, "random_forest_model")
        return rf_model

    def evaluate_model(self, model: PipelineModel, test_data: DataFrame) -> Dict[str, float]:
        """
        Evaluates the model using F1 Score, Precision, Recall, and ROC AUC on the test dataset.

        :param model: Trained RandomForest model.
        :param test_data: Test dataset for evaluation.
        :return: Dictionary with evaluation metrics (F1 Score, Precision, Recall, ROC AUC).
        """
        predictions = model.transform(test_data)
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                         metricName="f1")
        f1_score = evaluator_f1.evaluate(predictions)
        evaluator_precision = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                                metricName="weightedPrecision")
        precision_score = evaluator_precision.evaluate(predictions)
        evaluator_recall = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                             metricName="weightedRecall")
        recall_score = evaluator_recall.evaluate(predictions)
        evaluator_auc = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction",
                                                      metricName="areaUnderROC")
        roc_auc_score = evaluator_auc.evaluate(predictions)

        # Log metrics to MLflow
        mlflow.log_metric("F1 Score", f1_score)
        mlflow.log_metric("Precision", precision_score)
        mlflow.log_metric("Recall", recall_score)
        mlflow.log_metric("ROC AUC", roc_auc_score)

        return {
            "F1 Score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "ROC AUC": roc_auc_score
        }

    def predict(self, model: PipelineModel, text: str) -> float:
        """
        Predicts the probability of the label for a given input text.

        :param model: Trained RandomForest model.
        :param text: Input text string for prediction.
        :return: Predicted probability for the input text.
        """
        tokenized_text = self.tokenizer.transform(self.spark.createDataFrame([(text,)], ["Content"]))
        filtered_text = self.remover.transform(tokenized_text)

        lemmatize_udf = udf(lambda words: self.lemmatize_words(words), ArrayType(StringType()))
        lemmatized_text = filtered_text.withColumn("lemmatized_words", lemmatize_udf(col("filtered_words")))

        featurized_text = self.hashing_tf.transform(lemmatized_text)
        idf_model = self.idf.fit(featurized_text)
        rescaled_text = idf_model.transform(featurized_text)

        prediction = model.transform(rescaled_text)
        probability = prediction.select("probability").collect()[0][0]

        return probability


if __name__ == "__main__":
    mlflow.set_experiment("text_classification")

    with mlflow.start_run():
        trainer = Trainer(data_path="../data/train_dataset.csv")
        train_data, test_data = trainer.preprocessing()
        model = trainer.train_model(train_data)
        metrics = trainer.evaluate_model(model, test_data)

        print(f"Metrics: {metrics}")

        text = "This is an example of a hateful message"
        prediction = trainer.predict(model, text)
        print(f"Prediction: {prediction}")
