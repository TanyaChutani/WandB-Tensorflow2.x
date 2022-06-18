import tensorflow as tf


class Classification(tf.keras.models.Model):
    def __init__(self, num_classes=10, width=32, height=32):
        super(Classification, self).__init__()
        self.width = tf.constant(width, dtype=tf.int32)
        self.height = tf.constant(height, dtype=tf.int32)
        self.num_classes = tf.constant(num_classes, dtype=tf.int32)
        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=((self.height, self.width, 3)),
            pooling="max",
            classes=self.num_classes,
        )
        self.output_layer = tf.keras.layers.Dense(self.num_classes)

    def compile(self, optimizer, metric, loss):
        super(Classification, self).compile()
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer

    def call(self, input_tensor):
        x = self.resnet(input_tensor)
        x = self.output_layer(x)
        return x

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:

            model_output = self(images, training=True)
            loss = tf.reduce_mean(self.loss(labels, model_output))
            acc = self.metric(labels, model_output)

        gradients = tape.gradient(loss, self.trainable_variables)
        return {"loss": loss, "accuracy": acc}

    def test_step(self, data):
        images, labels = data
        model_output = self(images, training=False)
        loss = tf.reduce_mean(self.loss(labels, model_output))
        acc = self.metric(labels, model_output)
        return {"loss": loss, "accuracy": acc}

    def get_config(self):
        return {"num_classes": self.num_classes}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
