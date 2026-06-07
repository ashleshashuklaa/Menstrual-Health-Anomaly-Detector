import flwr as fl

from flower_client import FlowerClient


def client_fn(context):

    client_id = context.node_config["partition-id"]

    return FlowerClient(client_id)


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=50,
    min_available_clients=50,
)


if __name__ == "__main__":

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=50,
        config=fl.server.ServerConfig(
            num_rounds=20
        ),
        strategy=strategy,
    )