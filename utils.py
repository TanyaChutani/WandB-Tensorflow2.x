import wandb

def upload_dataset_wb(dataset, name, config):
    table = wandb.Table(columns=["id", "images", "classes"])

    for idx, (img, label) in enumerate(dataset):
        for i in range(config.batch_size):
            table.add_data(
                (idx + 1) * config.batch_size,
                wandb.Image(np.array(img[i])),
                label[i][0],
            )

    artifact = wandb.Artifact(
        name=name,
        type="dataset",
        metadata={
            "source": "keras.dataset.mnist",
            "size": len(dataset) * config.batch_size,
        },
    )
    artifact.add(table, name)
    return artifact


def download_dataset_wb(run, artifact_name):
    raw_artifact = run.use_artifact(artifact_name)
    dataset = raw_artifact.download()
    return dataset

def save_model_artifact(model, name, config):
    model_artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=dict(config),
        description="ResNet50 trained on cifar10 dataset",
    )
    model.save("resnet50.keras")
    model_artifact.add_file("resnet50.keras")
    wandb.save("resnet50.keras")
    return model_artifact
