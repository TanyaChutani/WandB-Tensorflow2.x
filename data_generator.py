import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        mode,
        batch_size,
        resize_dim,
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.__load_dataset()
        self.on_epoch_end()

    def __load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.images, self.labels = (
            (x_train, y_train) if self.mode == "train" else (x_test, y_test)
        )

    def __len__(self):
        return (self.images).shape[0] // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_keys_temp = [self.images[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        images = self.__preprocess(image_keys_temp)
        return images, labels

    def on_epoch_end(self):
        self.indexes = tf.range((self.images).shape[0])
        if self.mode == "train":
            tf.random.shuffle(self.indexes)

    def __preprocess(self, image):
        image = tf.image.resize(
            image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )
        image = image / 255.0
        image = tf.cast(image, tf.float32)
        return image
