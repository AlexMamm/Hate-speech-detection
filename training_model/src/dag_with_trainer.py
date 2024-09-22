import uuid
from datetime import datetime, timedelta
from airflow import DAG, settings
from airflow.models import Connection, Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.yandex.operators.yandexcloud_dataproc import (
    DataprocCreateClusterOperator,
    DataprocCreatePysparkJobOperator,
    DataprocDeleteClusterOperator,
)


YC_DP_FOLDER_ID = 'b1gt8j1njal2lvge27fp'
YC_DP_SUBNET_ID = 'e2l2bk3ce3jdmcs41vvg'
YC_DP_SA_ID = 'aje1spknjub3fd9orcbt'
YC_DP_AZ = 'ru-central1-b'
YC_DP_SSH_PUBLIC_KEY = Variable.get("SSH_PUBLIC")
YC_DP_GROUP_ID = 'enpsr5k5crdqaq3a5jrp'

YC_INPUT_DATA_BUCKET = 'amamylov-mlops/airflow/'
YC_SOURCE_BUCKET = 'amamylov-mlops'
YC_DP_LOGS_BUCKET = 'amamylov-mlops/airflow_logs/'
MLFLOW_S3_ENDPOINT_URL = 'https://storage.yandexcloud.net/'

session = settings.Session()
ycS3_connection = Connection(
    conn_id='yc-s3',
    conn_type='s3',
    host='https://storage.yandexcloud.net/',
    extra={
        "aws_access_key_id": Variable.get("S3_KEY_ID"),
        "aws_secret_access_key": Variable.get("S3_SECRET_KEY"),
        "host": MLFLOW_S3_ENDPOINT_URL
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycS3_connection.conn_id).first():
    session.add(ycS3_connection)
    session.commit()

ycSA_connection = Connection(
    conn_id='yc-SA',
    conn_type='yandexcloud',
    extra={
        "extra__yandexcloud__public_ssh_key": Variable.get("DP_PUBLIC_SSH_KEY"),
        "extra__yandexcloud__service_account_json_path": Variable.get("DP_SA_PATH")
    }
)

if not session.query(Connection).filter(Connection.conn_id == ycSA_connection.conn_id).first():
    session.add(ycSA_connection)
    session.commit()

with DAG(
        dag_id='train_model',
        start_date=datetime(year=2024, month=1, day=20),
        schedule_interval=timedelta(hours=6),
        catchup=False
) as ingest_dag:

    create_spark_cluster = DataprocCreateClusterOperator(
        task_id='dp-cluster-create-task',
        folder_id=YC_DP_FOLDER_ID,
        cluster_name=f'tmp-dp-{uuid.uuid4()}',
        cluster_description='Temporary cluster for Spark processing under Airflow orchestration',
        subnet_id=YC_DP_SUBNET_ID,
        s3_bucket=YC_DP_LOGS_BUCKET,
        service_account_id=YC_DP_SA_ID,
        ssh_public_keys=YC_DP_SSH_PUBLIC_KEY,
        zone=YC_DP_AZ,
        cluster_image_version='2.0.43',
        masternode_resource_preset='s3-c2-m8',
        masternode_disk_type='network-ssd',
        masternode_disk_size=20,
        datanode_resource_preset='s3-c4-m16',
        datanode_disk_type='network-ssd',
        datanode_disk_size=20,
        datanode_count=2,
        services=['YARN', 'SPARK', 'HDFS', 'MAPREDUCE'],
        computenode_count=0,
        connection_id=ycSA_connection.conn_id,
        dag=ingest_dag
    )

    poke_spark_processing = DataprocCreatePysparkJobOperator(
        task_id='dp-cluster-pyspark-task',
        main_python_file_uri=f's3a://{YC_SOURCE_BUCKET}/hate_speech_detection/trainer.py',
        connection_id=ycSA_connection.conn_id,
        dag=ingest_dag,
        properties={'spark.submit.deployMode': 'cluster',
                    'spark.yarn.dist.archives': f's3a://{YC_SOURCE_BUCKET}/hate_speech_detection/train_model_venv.tar.gz#venv1',
                    'spark.yarn.appMasterEnv.PYSPARK_PYTHON': './venv1/bin/python',
                    'spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON': './venv1/bin/python',
                    'spark.yarn.appMasterEnv.AWS_ACCESS_KEY_ID': Variable.get("S3_KEY_ID"),
                    'spark.yarn.appMasterEnv.AWS_SECRET_ACCESS_KEY': Variable.get("S3_SECRET_KEY"),
                    'spark.yarn.appMasterEnv.MLFLOW_S3_ENDPOINT_URL': MLFLOW_S3_ENDPOINT_URL}
    )

    delete_spark_cluster = DataprocDeleteClusterOperator(
        task_id='dp-cluster-delete-task',
        trigger_rule=TriggerRule.ALL_DONE,
        dag=ingest_dag
    )

    create_spark_cluster >> poke_spark_processing >> delete_spark_cluster
