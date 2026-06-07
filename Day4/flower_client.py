class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id):
        ...
        self.model = build_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):

        self.model.set_weights(parameters)

        self.model.fit(
            self.X,
            self.X,
            epochs=5,
            batch_size=16,
            verbose=0
        )

        return (
            self.model.get_weights(),
            len(self.X),
            {}
        )

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)

        loss = self.model.evaluate(
            self.X,
            self.X,
            verbose=0
        )

        return loss, len(self.X), {}