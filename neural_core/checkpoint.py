import json


class Checkpoint:
    def __init__(self, model_name="jarvis_model"):
        self.model_name = model_name

    def create(self, parameters):
        return {
            "model_name": self.model_name,
            "parameters": parameters,
        }

    def save(self, checkpoint, filepath):
        with open(filepath, "w") as file:
            json.dump(checkpoint, file, indent=4)

    def load(self, filepath):
        with open(filepath, "r") as file:
            return json.load(file)