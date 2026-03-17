from kfp import compiler
from pipelines.fraud_pipeline import fraud_pipeline

compiler.Compiler().compile(
    pipeline_func=fraud_pipeline,
    package_path="fraud_pipeline.yaml"
)