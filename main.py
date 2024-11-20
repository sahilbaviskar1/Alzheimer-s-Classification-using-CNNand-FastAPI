import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("AlzheimerModel")
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    logger.debug(f"Device: {tpu.master()}")
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception as e:
    logger.warning(f"TPU not found, using default strategy. Error: {str(e)}")
    strategy = tf.distribute.get_strategy()
logger.info(f"Number of replicas: {strategy.num_replicas_in_sync}")

logger.info(f"TensorFlow Version: {tf.__version__}")

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [176, 208]
EPOCHS = 55
NUM_CLASSES = 4
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
train_ds = None
val_ds = None
model = None

def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train_model():
    global train_ds, val_ds, model
    logger.debug("Preparing the datasets.")
    try:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "dataset/train",
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "dataset/train",
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
        train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    except Exception as e:
        logger.error(f"Error while loading datasets: {str(e)}")
        raise

    with strategy.scope():
        model = build_model()
        METRICS = [tf.keras.metrics.AUC(name='auc')]
        model.compile(
            optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=METRICS
        )
    
    logger.debug("Starting model training.")
    try:
        exponential_decay_fn = exponential_decay(0.01, 20)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.keras", save_best_only=True)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
            epochs=EPOCHS
        )
        return history
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

@app.get("/")
def read_root():
    return {"message": "Welcome to the Alzheimer's Disease Classification API"}

@app.post("/train")
def train():
    try:
        history = train_model()
        return JSONResponse(content={"message": "Model training completed", "history": history.history})
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Model training failed.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        try:
            logger.debug("Loading the model.")
            model = load_model("alzheimer_model.keras")
        except Exception as e:
            logger.error(f"Failed to load the model: {str(e)}")
            raise HTTPException(status_code=500, detail="Model loading failed.")
    
    try:
        logger.debug(f"Processing the uploaded image: {file.filename}")
        img = PIL.Image.open(file.file).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        logger.debug("Making predictions.")
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]

        logger.debug(f"Prediction successful: {predicted_class}")
        return {"filename": file.filename, "prediction": predicted_class}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail="Prediction failed.")

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
