from azure.ai.ml import MLClient
from pathlib import Path


def load_all(client: MLClient):
    """Load all Azure ML Designer components.

    Parameters
    ----------
    client : MLClient
        Azure ML workspace client.
    """
    for component in Path(__file__).parent.glob("*.y*ml"):
        client.components.create_or_update(component)
