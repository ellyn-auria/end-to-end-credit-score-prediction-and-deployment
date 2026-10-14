import os
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

local_pipeline_session = LocalPipelineSession()

try:
    role = sagemaker.get_execution_role()
except Exception:
    role = "LabRole"

INSTANCE_TYPE = "local"
FRAMEWORK_VER = "1.2-1"
BASE_PATH = "/home/ec2-user/SageMaker"
DATA_DIR = f"{BASE_PATH}/data"
SRC_DIR = f"{BASE_PATH}/src"

LOCAL_OUTPUT = f"file://{BASE_PATH}"

for sub in ["ingested", "train", "test", "model", "eval"]:
    os.makedirs(f"{BASE_PATH}/{sub}", exist_ok=True)

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Folder data tidak ditemukan: {DATA_DIR}")
if not os.path.isdir(SRC_DIR):
    raise FileNotFoundError(
        f"Folder src tidak ditemukan: {SRC_DIR}")

sklearn_processor = SKLearnProcessor(
    framework_version = FRAMEWORK_VER,
    role = role,
    instance_type = INSTANCE_TYPE,
    instance_count = 1,
    sagemaker_session = local_pipeline_session,
)

sklearn_estimator = SKLearn(
    entry_point = "train.py",
    source_dir = SRC_DIR,
    role = role,
    instance_type = INSTANCE_TYPE,
    framework_version = FRAMEWORK_VER,
    sagemaker_session = local_pipeline_session,
    hyperparameters = {"model-names": "XGBoost,RandomForest,DecisionTree"},
)

step_ingest = ProcessingStep(
    name = "CreditIngest",
    processor = sklearn_processor,
    inputs = [
        ProcessingInput(
            source = DATA_DIR,
            destination = "/opt/ml/processing/input",
        )
    ],
    outputs = [
        ProcessingOutput(
            output_name = "ingested",
            source = "/opt/ml/processing/ingested",
            destination = f"{LOCAL_OUTPUT}/ingested",
        )
    ],
    code = os.path.join(SRC_DIR, "data_ingestion.py"),
)

step_preprocess = ProcessingStep(
    name = "CreditPreprocess",
    processor = sklearn_processor,
    inputs = [
        ProcessingInput(
            source = step_ingest.properties.ProcessingOutputConfig
                              .Outputs["ingested"].S3Output.S3Uri,
            destination = "/opt/ml/processing/ingested",
        )
    ],
    outputs = [
        ProcessingOutput(
            output_name = "train",
            source = "/opt/ml/processing/train",
            destination = f"{LOCAL_OUTPUT}/train",
        ),
        ProcessingOutput(
            output_name = "test",
            source = "/opt/ml/processing/test",
            destination = f"{LOCAL_OUTPUT}/test",
        ),
    ],
    code = os.path.join(SRC_DIR, "preprocessing.py"),
)

step_train = TrainingStep(
    name = "CreditTrain",
    estimator = sklearn_estimator,
    inputs = {
        "train": TrainingInput(
            s3_data = step_preprocess.properties.ProcessingOutputConfig
                          .Outputs["train"].S3Output.S3Uri,
        )
    },
)

evaluation_report = PropertyFile(
    name = "EvaluationReport",
    output_name = "evaluation",
    path = "evaluation.json",
)

step_eval = ProcessingStep(
    name = "CreditEval",
    processor = sklearn_processor,
    inputs = [
        ProcessingInput(
            source = step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination = "/opt/ml/processing/model",
        ),
        ProcessingInput(
            source = step_preprocess.properties.ProcessingOutputConfig
                              .Outputs["test"].S3Output.S3Uri,
            destination = "/opt/ml/processing/test",
        ),
    ],
    outputs = [
        ProcessingOutput(
            output_name = "evaluation",
            source = "/opt/ml/processing/evaluation",
            destination = f"{LOCAL_OUTPUT}/eval",
        )
    ],
    code = os.path.join(SRC_DIR, "evaluation.py"),
    property_files = [evaluation_report],
)

pipeline = Pipeline(
    name = "CreditScore-Local-Pipeline",
    steps = [step_ingest, step_preprocess, step_train, step_eval],
    sagemaker_session = local_pipeline_session,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("\nPipeline local execution started.")