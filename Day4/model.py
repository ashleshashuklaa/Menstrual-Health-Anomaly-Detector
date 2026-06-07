from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)

def build_model():

    inputs = Input(shape=(6, 6))

    # Encoder
    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)

    # Bottleneck
    latent = Dense(16, activation="relu")(x)

    # Decoder
    x = RepeatVector(6)(latent)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)

    outputs = TimeDistributed(Dense(6))(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model