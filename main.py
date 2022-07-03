import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-b", "--batch_size", type=int, default=24)
    parser.add_argument("-b", "--epoch", type=int, default=2)
    parser.add_argument("-b", "--image_size", type=int, default=32)
    args = parser.parse_args()
    return args


def train(
    config={
        "learning_rate": 0.0001,
        "epoch": 2,
        "batch_size": 24,
        "dim": 32,
        "architecture": "resnet50",
        "dataset": "CIFAR-10",
        "num_classes": 10,
    }
):
    with wandb.init(
        project="classification", config=config, job_type="load-data"
    ) as run:
        config = wandb.config
        train_dataset = DataGenerator(
            mode="train",
            batch_size=config.batch_size,
            resize_dim=(config.dim, config.dim),
        )

        train_artifact = upload_dataset_wb(train_dataset, "train_dataset", config)
        run.log_artifact(train_artifact, aliases="latest")

        test_dataset = DataGenerator(
            mode="test",
            batch_size=config.batch_size,
            resize_dim=(config.dim, config.dim),
        )

        test_artifact = upload_dataset_wb(test_dataset, "test_dataset", config)
        run.log_artifact(test_artifact, aliases="latest")

        #model = Classification()
        #model.build((config.batch_size, config.dim, config.dim, 3))
        model = get_model(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metric=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=config.epoch,
            callbacks=[
                WandbCallback(
                    monitor="val_loss",
                    log_weights=True,
                )
            ],
        )
        model.save(os.path.join(wandb.run.dir, "resnet_model"))
        model_artifact = save_model_artifact(model, "resnet50_model", config)
        run.log_artifact(model_artifact)

def main():
    args = parse_args()
    train()
    
if __name__ == "__main__":
    main()
