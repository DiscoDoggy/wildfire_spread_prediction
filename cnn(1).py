import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, Dropout, MaxPool2D, UpSampling2D

#initialize feature names for tfrecord parsing
INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph','pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
OUTPUT_FEATURES = ['FireMask']
FEATURE_VECTOR_SIZE = 64

#declare read in format for tf record parsing per feature
feature_descriptions = {
    'elevation' : tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'th' : tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'vs' : tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'tmmn' : tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'tmmx': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'sph': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'pr': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'pdsi': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'NDVI': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'population': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'erc': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'PrevFireMask': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32),
    'FireMask': tf.io.FixedLenFeature([FEATURE_VECTOR_SIZE, FEATURE_VECTOR_SIZE], tf.float32)
}

#tfrecord files to use for training data
train_dataset = tf.data.TFRecordDataset(["./next_day_wildfire_spread_train_00.tfrecord",
                                       "./next_day_wildfire_spread_train_01.tfrecord",
                                       "./next_day_wildfire_spread_train_02.tfrecord", 
                                       "./next_day_wildfire_spread_train_03.tfrecord", 
                                       "./next_day_wildfire_spread_train_04.tfrecord",
                                         "./next_day_wildfire_spread_train_05.tfrecord",
                                        "./next_day_wildfire_spread_train_06.tfrecord", 
                                        "./next_day_wildfire_spread_train_07.tfrecord", "./next_day_wildfire_spread_train_08.tfrecord", "./next_day_wildfire_spread_train_09.tfrecord", "./next_day_wildfire_spread_train_10.tfrecord"])

#tf record data to use for validaiton
validation_dataset = tf.data.TFRecordDataset([
    './next_day_wildfire_spread_eval_00.tfrecord',
    './next_day_wildfire_spread_eval_01.tfrecord'
])

#given a sample from a tf record, read in the data according to the feature descriptions declared above
def parse_dataset(data_sample):
    features = tf.io.parse_single_example(data_sample, feature_descriptions)
    inputs_list = [features.get(key) for key in INPUT_FEATURES]

    stacked_inputs = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(stacked_inputs, [1,2,0])

    print(input_img.shape)

    outputs_list = [features.get(key) for key in OUTPUT_FEATURES]

    assert outputs_list, 'outputs_list should not be empty'
    
    outputs_stacked = tf.stack(outputs_list, axis=0)
    output_img = tf.transpose(outputs_stacked, [1,2,0])
    output_img = tf.cast(output_img > 0.5, tf.float32)
    
    return input_img, output_img

#for tf records get samples by calling parse_dataset
def get_dataset(dataset_tfrecords):

    dataset = dataset_tfrecords.map(
        lambda x : parse_dataset(x)
    )
    
    return dataset

#create training and validation set
training_set = get_dataset(train_dataset)
validation_set = get_dataset(validation_dataset)
training_batches = training_set.batch(32)
val_batches = validation_set.batch(32)

#build model
autoencoder = tf.keras.Sequential([
    Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=(64,64,12)),
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    Conv2D(filters=16, kernel_size=(3,3), padding='same', strides=(1,1)),

    #main block one
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    MaxPool2D(2,2),
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    Conv2D(filters=16, kernel_size=(3,3), padding='same', strides=(1,1)),
    # Dropout(rate=0.1),

    #main block 2
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    MaxPool2D(2,2),
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    Conv2D(filters=32, kernel_size=(3,3), padding='same', strides=(1,1)),
    # Dropout(rate=0.1),

    #upsample
    UpSampling2D(),

    #main block 3
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    # MaxPool2D(2,2),
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    Conv2D(filters=32, kernel_size=(3,3), padding='same', strides=(1,1)),
    # Dropout(rate=0.1),

    UpSampling2D(),

    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    # MaxPool2D(2,2),
    LeakyReLU(alpha=0.1),
    # Dropout(rate=0.1),
    Conv2D(filters=16, kernel_size=(3,3), padding='same', strides=(1,1)),
    # Dropout(rate=0.1),

    #fin
    Conv2D(filters=1, kernel_size=(3,3), padding='same', strides=(1,1))
])

#implement a weighted loss as per research paper
def weighted_loss(pos_weight):
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_weight
        ))
    return loss_fn

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder.compile(
    optimizer=optimizer,
    loss=weighted_loss(20.0),
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

history = autoencoder.fit(training_batches, validation_data=val_batches, epochs=200, steps_per_epoch=340)

test_dataset = tf.data.TFRecordDataset(['./next_day_wildfire_spread_eval_00.tfrecord'])
test_set = get_dataset(test_dataset)

NUM_SAMPLES = 5
predicted_maps = []
ground_truth_maps = []

#testing NUM_SAMPLES 
for sample, label in test_set:
    #collapse outer dimension and check if this sample contains at least one fire pixel
    label_np = np.squeeze(label.numpy())
    if np.any(label_np == 1):
        sample_batch = np.expand_dims(sample, axis=0)
        pred = autoencoder.predict(sample_batch)
        prob = tf.sigmoid(pred)[0, :, :, 0]
        output_image = tf.cast(prob > 0.5, tf.float32)

        predicted_maps.append(output_image.numpy())
        ground_truth_maps.append(label_np)

    if len(predicted_maps) == NUM_SAMPLES:
        break


# Plotting -- save output as image

fig, axes = plt.subplots(nrows=2, ncols=NUM_SAMPLES, figsize=(NUM_SAMPLES * 3, 6))

for i in range(NUM_SAMPLES):
    # Ground Truth
    axes[0, i].imshow(ground_truth_maps[i], cmap='viridis', vmin=0, vmax=1)
    axes[0, i].set_title(f"Ground Truth {i+1}")
    axes[0, i].axis('off')

    # Prediction
    axes[1, i].imshow(predicted_maps[i], cmap='viridis', vmin=0, vmax=1)
    axes[1, i].set_title(f"Prediction {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("fire_map.png", dpi=300)
plt.close()
print("hello world")