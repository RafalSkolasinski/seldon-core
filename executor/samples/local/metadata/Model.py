import logging
import os


class Model:
    def predict(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        return features.tolist()

    def init_metadata(self):

        meta = {
            "name": "model-name",
            "versions": ["model-version"],
            "platform": "platform-name",
            "inputs": [{"name": "input", "datatype": "BYTES", "shape": [3]}],
            "outputs": [{"name": "output", "datatype": "BYTES", "shape": [5, 6]}],
        }

        return meta
