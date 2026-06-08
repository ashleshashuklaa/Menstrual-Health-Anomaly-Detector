import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.metrics import f1_score, roc_auc_score

X_train = np.load("../Day2/X_train.npy")
y_train = np.load("../Day2/y_train.npy")
X_test  = np.load("../Day2/X_test.npy")
y_test  = np.load("../Day2/y_test.npy")

X_train_normal = X_train[y_train == 0]

def build_model(enc1, enc2, latent):
    inputs = Input(shape=(6, 6))
    x = LSTM(enc1, return_sequences=True)(inputs)
    x = LSTM(enc2)(x)
    x = Dense(latent, activation="relu")(x)
    x = RepeatVector(6)(x)
    x = LSTM(enc2, return_sequences=True)(x)
    x = LSTM(enc1, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(6))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

configs = [
    ("Small",   32,  16,  8),
    ("Current", 64,  32, 16),
    ("Large",  128,  64, 32),
]

print(f"{'Config':<10} {'AUC':>8} {'F1':>8} {'Time(s)':>10}")
print("-" * 40)

for name, enc1, enc2, latent in configs:
    start = time.time()
    model = build_model(enc1, enc2, latent)
    model.fit(X_train_normal, X_train_normal,
              epochs=30, batch_size=16,
              validation_split=0.1, verbose=0)
    elapsed = time.time() - start

    recon_train = model.predict(X_train_normal, verbose=0)
    errors_train = np.mean(np.square(X_train_normal - recon_train), axis=(1,2))
    threshold = errors_train.mean() + 2 * errors_train.std()

    recon_test = model.predict(X_test, verbose=0)
    errors_test = np.mean(np.square(X_test - recon_test), axis=(1,2))
    preds = (errors_test > threshold).astype(int)

    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, errors_test)

    print(f"{name:<10} {auc:>8.4f} {f1:>8.4f} {elapsed:>10.1f}")