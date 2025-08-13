import tensorflow as tf
from tensorflow.keras import layers


class DeepTFactor(tf.keras.Model):
    def __init__(self, out_features=[0]):
        super(DeepTFactor, self).__init__()
        self.explainECs = out_features
        self.layer_info = [[4, 4, 16], [12, 8, 4], [16, 4, 4]]
        self.cnn0 = CNN(self.layer_info)
        self.fc1 = layers.Dense(512)
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(len(out_features))
        self.bn2 = layers.BatchNormalization()
        self.out_act = layers.Activation('sigmoid')
        self.relu = layers.Activation('relu')

    def call(self, x):
        #x = tf.cast(x, tf.float32)
        print("X shape 0: ", x.shape)
        x = self.cnn0(x)
        x = tf.keras.layers.Flatten()(x)
        print("X shape 1: ", x.shape)
        #x = tf.reshape(x, (x.shape[0], -1))
        x = self.relu(self.bn1(self.fc1(x)))
        print("X shape 2: ", x.shape)
        x = self.out_act(self.bn2(self.fc2(x)))
        print("X shape 3: ", x.shape)
        x = tf.reshape(x, (-1, self.fc2.units))
        print("X shape 4: ", x.shape)
        return x


class CNN(tf.keras.Model):
    def __init__(self, layer_info):
        super(CNN, self).__init__()
        self.relu = layers.Activation('relu')
        self.dropout = layers.Dropout(0.1)
        self.layers_list = []
        pooling_sizes = []

        for subnetwork in layer_info:
            pooling_size = 0
            self.layers_list.append(self.make_subnetwork(subnetwork))
            for kernel in subnetwork:
                pooling_size += (-kernel + 1)
            pooling_sizes.append(pooling_size)

        if len(set(pooling_sizes)) != 1:
            raise "Different kernel sizes between subnetworks"

        pooling_size = pooling_sizes[0]
        num_subnetwork = len(layer_info)

        self.conv = layers.Conv2D(128 * 3, (1, 1))
        self.batchnorm = layers.BatchNormalization()
        self.pool = layers.MaxPool2D((1000 + pooling_size, 1), strides=1)

    def make_subnetwork(self, subnetwork):
        subnetworks = []

        for i, kernel in enumerate(subnetwork):
            if i == 0:
                subnetworks.append(layers.Conv2D(128, (kernel, 22)))
                subnetworks.append(layers.BatchNormalization())
                subnetworks.append(layers.Activation('relu'))
                subnetworks.append(layers.Dropout(0.1))
            else:
                subnetworks.append(layers.Conv2D(128, (kernel, 1)))
                subnetworks.append(layers.BatchNormalization())
                subnetworks.append(layers.Activation('relu'))
                subnetworks.append(layers.Dropout(0.1))

        return tf.keras.Sequential(subnetworks)

    def call(self, x):
        xs = []

        for layer in self.layers_list:
            xs.append(layer(x))

        x = tf.concat(xs, axis=-1)
        x = self.relu(self.batchnorm(self.conv(x)))
        x = self.pool(x)

        return x
