from fastapi import FastAPI, HTTPException, Depends
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
import uvicorn

from config import Config
from api_schemas import CommentRequest, CommentPrediction


config = Config()
app = FastAPI()

spark: SparkSession = SparkSession.builder \
    .appName("Text Classification with PySpark") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "2") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.storage.memoryFraction", "0.6") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()


def load_model() -> PipelineModel:
    """
    Loads the machine learning model required for predictions.
    Downloads necessary NLTK resources (wordnet and omw-1.4) for text processing.

    :return: Loaded PySpark PipelineModel.
    :raises HTTPException: If the model cannot be loaded.
    """
    try:
        model = PipelineModel.load(config.model_path)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Model could not be loaded")


@app.get("/api/v1/health")
async def health_check() -> dict:
    """
    Health check endpoint to ensure the service is up and running.

    :return: JSON object indicating the service is healthy.
    """
    return {"status": "healthy"}


@app.get("/api/v1/ready")
async def readiness_check() -> dict:
    """
    Readiness check to determine if the model is ready for predictions.

    :return: JSON object indicating whether the model is ready.
    :raises HTTPException: If the model cannot be loaded.
    """
    try:
        model = load_model()
        if model:
            return {"status": "ready"}
        else:
            return {"status": "not ready"}
    except Exception:
        return {"status": "not ready"}


@app.get("/api/v1/startup")
async def startup_check() -> dict:
    """
    Checks if the Spark session has started successfully.

    :return: JSON object indicating whether the Spark session is started.
    """
    try:
        if spark:
            return {"status": "started"}
        else:
            return {"status": "not started"}
    except Exception:
        return {"status": "not started"}


@app.post("/api/v1/predict", response_model=CommentPrediction)
async def predict(request: CommentRequest, model: PipelineModel = Depends(load_model)) -> CommentPrediction:
    """
    Predicts the likelihood of a comment being toxic.

    :param request: Input text to be analyzed.
    :param model: Loaded PySpark model for making predictions.
    :return: Prediction result including probability and label (toxic or normal).
    :raises HTTPException: If there is an error during prediction.
    """
    try:
        tokenized_text: DataFrame = spark.createDataFrame([(request.text,)], ["Content"])
        processed_text = model.transform(tokenized_text)

        normal_probability: float = round(processed_text.select("probability").collect()[0][0][0], 3)
        hate_probability: float = round(processed_text.select("probability").collect()[0][0][1], 3)
        label: str = "Toxic comment" if hate_probability > 0.5 else "Normal comment"
        return CommentPrediction(
            normal_probability=normal_probability,
            hate_probability=hate_probability,
            label=label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        host=config.backend_host,
        port=config.backend_port,
    )
