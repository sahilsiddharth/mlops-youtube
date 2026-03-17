from kfp import dsl


from kfp.kubernetes import set_image_pull_policy

@dsl.container_component
def train_model():
    return dsl.ContainerSpec(
        image="fraud-mlops:v1",
        command=["python"],
        args=["scripts/train_model.py"]
    )

@dsl.pipeline(
    name="fraud-detection-training-pipeline",
    description="Train fraud detection model"
)
def fraud_pipeline():
    train_model_task = train_model()
    
    # Tell k8s not to pull from registry
    set_image_pull_policy(train_model_task, "IfNotPresent")