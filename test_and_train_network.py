from image_proccessor import load_image, load_labels
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D, MaxPooling2D, Flatten, Add, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

vgg = VGG16(include_top=False)
vgg.summary()

# Building VGG neural network
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    # Feature extraction using VGG16
    vgg = VGG16(include_top=False)(input_layer)

    # Classification branch 
    class_f = GlobalMaxPooling2D()(vgg)
    class_dense = Dense(2048, activation='relu')(class_f)
    class_output = Dense(1, activation='sigmoid', name='classification')(class_dense)
    
    # Bounding box branch
    regress_f = GlobalMaxPooling2D()(vgg)
    regress_dense = Dense(2048, activation='relu')(regress_f)
    regress_output = Dense(4, activation='sigmoid', name='regression')(regress_dense)
    
    facetracker = Model(inputs=input_layer, outputs=[class_output, regress_output])
    return facetracker

# load image and create tensorflow dataset
train_images = tf.data.Dataset.list_files('augmented_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)

test_images = tf.data.Dataset.list_files('augmented_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)

val_images = tf.data.Dataset.list_files('augmented_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)

# load labels and create tensorflow dataset
train_labels = tf.data.Dataset.list_files('augmented_data\\train\\labels\\*.json', shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('augmented_data\\test\\labels\\*.json', shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('augmented_data\\val\\labels\\*.json', shuffle = False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

print(train_labels.as_numpy_iterator().next())
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

# Combining labels and images together and shuffling them.
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(8000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(2000)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1800)
val = val.batch(8)
val = val.prefetch(4)

# localization loss function
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

# Define a learning rate schedule with ExponentialDecay
batches_per_epoch = len(train)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate, decay_steps = batches_per_epoch, decay_rate = 0.75, staircase = False)

opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# Custom Face Model
class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

# Build the model and compile
facetracker = build_model()
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

# Log and train the model
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# Visualize the performance of the model
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

# Plot total loss
ax[0].plot(hist.history['total_loss'], color='blue', label='Training Loss')
ax[0].plot(hist.history['val_total_loss'], color='red', label='Validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].title.set_text('Total Loss')
ax[0].legend()

# Plot classification loss
ax[1].plot(hist.history['class_loss'], color='blue', label='Training Class Loss')
ax[1].plot(hist.history['val_class_loss'], color='red', label='Validation Class Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

# Plot regression loss
ax[2].plot(hist.history['regress_loss'], color='blue', label='Training Regression Loss')
ax[2].plot(hist.history['val_regress_loss'], color='red', label='Validation Regression loss')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# Test the model with predictions on test dataset.
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
y = facetracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = test_sample[0][idx]
    sample_coords = y[1][idx]
    
    if y[0][idx] > 0.9:
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)
    
    ax[idx].imshow(sample_image)
    plt.show()

# Save the model into a file
facetracker.save('cnn_face_detection_model.keras')

facetracker = load_model('cnn_face_detection_model.keras')

facetracker.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'classification': tf.keras.losses.BinaryCrossentropy(),
        'regression': localization_loss
    },
    metrics={
        'classification': [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives()
        ],
        'regression': localization_loss
    }
)

# Get the model accuracy
test_data = test.as_numpy_iterator()
true_labels = []
predicted_labels = []

for batch in test_data:
    X_test, y_test = batch
    predictions = facetracker.predict(X_test)
    true_labels.extend(y_test[0])  
    predicted_labels.extend(predictions[0])

# Convert the lists to numpy arrays for easier comparison
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Convert predicted probabilities to binary predictions
binary_predictions = (predicted_labels > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(true_labels == binary_predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")