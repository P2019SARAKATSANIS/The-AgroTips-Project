import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, metrics, backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet50
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# Define image dimensions, batch size and number of classes of the model.
img_height = 512
img_width = 512
batch_size = 32 #based you you symstem, it might vary
num_classes = 7

class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrixMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name='conf_matrix', shape=(num_classes, num_classes), initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert true and predicted labels to 1D
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=-1)

        # Compute confusion matrix for the batch
        batch_conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        # Update confusion matrix state (accumulated over all batches)
        self.confusion_matrix.assign_add(batch_conf_matrix)

    def result(self):
        # For simplicity, return the overall accuracy derived from the confusion matrix
        cm = self.confusion_matrix
        diagonal_sum = tf.reduce_sum(tf.linalg.diag_part(cm))  # Sum of diagonal values (true positives)
        total_sum = tf.reduce_sum(cm)  # Sum of all values (total predictions)

        return diagonal_sum / total_sum  # Return accuracy based on confusion matrix

    def reset_states(self):
        # Reset the confusion matrix at the start of each epoch
        self.confusion_matrix.assign(tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32))

    def print_confusion_matrix(self):
        # Convert the confusion matrix tensor to a NumPy array for better printing
        cm = self.confusion_matrix.numpy()
        np.set_printoptions(threshold=np.inf)  # Ensure full matrix is printed without truncation
        print("\nConfusion Matrix at the end of the epoch:\n")
        for row in cm:
            print(' '.join(f'{int(val):4}' for val in row))  # Print each row with proper spacing


class PrintConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, confusion_matrix_metric):
        super().__init__()
        self.confusion_matrix_metric = confusion_matrix_metric

    def on_epoch_end(self, epoch, logs=None):
        # Print the confusion matrix at the end of the epoch
        self.confusion_matrix_metric.print_confusion_matrix()





#This class, creates the f1score metric that is displayed when the model is running.

class F1Score(metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=1)
        y_pred = tf.cast(y_pred, tf.int32)

        # The True Positives, False Positivesa False Negatives are being calculated here.
        tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.equal(y_true, 1), self.dtype))
        fp = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred) & tf.equal(y_pred, 1), self.dtype))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred) & tf.equal(y_true, 1), self.dtype))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    #In this function, f1score is being calculated using the True Positives, False Positives and False Negatives
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Custom layer to apply changes to contrast
class RandomContrastCustom(tf.keras.layers.Layer):
    def __init__(self, lower, upper):
        super(RandomContrastCustom, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self, x):
        # Apply contrast in the range [lower, upper]
        contrast_factor = tf.random.uniform([], self.lower, self.upper)
        return tf.image.adjust_contrast(x, contrast_factor)

# Custom layer to apply changes in brightness
class RandomBrightnessCustom(tf.keras.layers.Layer):
    def __init__(self, lower, upper):
        super(RandomBrightnessCustom, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self, x):
      contrast_factor = tf.random.uniform([], self.lower, self.upper)
      return tf.image.adjust_brightness(x, contrast_factor)

    def call(self, x):
        # Apply brightness in the range [lower, upper]
        contrast_factor = tf.random.uniform([], self.lower, self.upper)
        return tf.image.adjust_brightness(x, contrast_factor)

# Data Augmentation
data_augmentation = keras.Sequential([
    RandomContrastCustom(0.4, 0.6),  # Apply contrast between 0.4 and 0.6
    RandomBrightnessCustom(0.4, 0.6)  # Randomly change the brightness by a factor of 0.2
])



# Load the training dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "thesis_images/",
    labels="inferred",
    label_mode="int",
    class_names= ['Class1-Healthy', 'Class2-Pseudomonas-Xanthomonas-Septoria', 'Class3-Alternaria', 'Class4-Cladosporium Leaf Mold', 'Class5-Downy Mildew', 'Class6-Ash Rot', 'Class7-Powdery Mildew'],
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
)


# Load the validation dataset
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "thesis_images/",
    labels="inferred",
    label_mode="int",
    class_names= ['Class1-Healthy', 'Class2-Pseudomonas-Xanthomonas-Septoria', 'Class3-Alternaria', 'Class4-Cladosporium Leaf Mold', 'Class5-Downy Mildew', 'Class6-Ash Rot', 'Class7-Powdery Mildew'],
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
)


# Data augmentation for the training dataset
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# Data augmentation for the validation dataset. We apply data augmentation to the validation also, because some images migh look common due to the fact that an identical flipped image might exist within the dataset. Using data augmentation, these images look less alike from the original ones and can be considered new. Also, we need those images within the range of 0.4 to 0.6 because even when running the model, images taken will be augmented to 55% brightness and contrast. This way, they give out more information.
val_data = val_data.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)


# Prefetch data for both training and validation sets
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# The ResNet50 model pre-trained on ImageNet, is being loaded here.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

#Here we freeze the base modeln in order to add the custom layers to it.
base_model.trainable = False

# Add new layers on top of the pre-trained base
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Adding the global average pooling layer
x = Dense(128, activation='relu')(x)  # Adding a fully connected layer
predictions = Dense(7, activation='softmax')(x)  # Adding the final classification layer with 7 classes


model = Model(inputs=base_model.input, outputs=predictions)

#We want the learning rate to start from 0.001. More than that would train the model too fast and the result will not be satisfying. Less than that, the model will train too slow, due to the callbacks, that can only be decreased and there is a chance that the model will never reach its full potential.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

confusion_matrix_metric = ConfusionMatrixMetric(num_classes=num_classes)


# Compile the model
model.compile(
    optimizer = optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', F1Score(), confusion_matrix_metric]
)



# The model summary is printed here
model.summary()



# Callbacks
#The model will stop early if the validation f1score does not increase. If the validation loss starts increasing or stays still (based on the lowest yet validation loss) for 3 times, the the learning rate decrease.
early_stopping = callbacks.EarlyStopping(monitor='val_f1_score', patience=5, restore_best_weights=True, mode='max')
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
confusion_matrix_callback = PrintConfusionMatrixCallback(confusion_matrix_metric)


# The model will train with 50 epoches, the split that we have give at validation and train dataset preprocessing and the callbacks we used.
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler, confusion_matrix_callback]
)

# Save the entire model
model.save('AgroTips_model.keras')
