from ConvNetFactory import ConvNetFactory
import tensorflow as tf
import keras
from keras.utils import np_utils
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="name of network to build")
ap.add_argument("-m", "--model", required=True, help="path to the output model file")
ap.add_argument("-d", "--dropout", type=int, default=-1, help="use dropout or not")
ap.add_argument("-f", "--activation", type=str, default="tanh", help="activation function to use (LeNet only)")
ap.add_argument("-e", "--epochs", type=int, default=32, help="number of epochs to train")
ap.add_argument("-b", "--batchsize", type=int, default=32, help="size of the batch to pass")
ap.add_argument("-v", "--verbose", type=int, default=1, help="verbosity level")
args = vars(ap.parse_args())

print("[INFO] Loading dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# normalize the data
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0
# vectorizing the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

kargs = {"dropout": args["dropout"] > 0, "activation": args["activation"]}

print("[INFO] Compiling the model...")
model = ConvNetFactory.build(args['network'], 3, 32, 32, 10, **kargs)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="best"+args["model"],
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy",
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode="auto",
    restore_best_weights=True)

print("[INFO] Training model...")
with tf.device('/GPU:0'):
    history = model.fit(x_train, y_train, batch_size=args['batchsize'], epochs=args['epochs'], verbose=args['verbose'],
                        callbacks=[early_stop, model_checkpoint_callback])

loss, accuracy = model.evaluate(x_test, y_test, batch_size=args['batchsize'], verbose=args['verbose'])
print(f"[INFO] Accuracy: {accuracy*100}%")

print("[INFO] Saving model...")
if not os.path.exists("models"):
    os.makedir("models")
model.save(f"models/{args['model']}")

print("[INFO] Saving metrics...")
with open("metrics.csv", "a") as f:
    f.write(f"{args['network']},{args['model']},{history.history['accuracy'][-1]},{round(accuracy*100,2)}%\n")
