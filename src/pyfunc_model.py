import json

import mlflow.pyfunc
import numpy as np
import torch


class RULModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.models import build_model

        with open(context.artifacts["params"]) as f:
            p = json.load(f)
        self.cap = p["rul_cap"]
        self.model = build_model(p["model"], p["n_features"], p["window"],
                                 hidden=p["hidden"], layers=p["layers"], dropout=p["dropout"])
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        self.model.eval()

    def predict(self, context, model_input, params=None):
        x = torch.as_tensor(np.asarray(model_input, dtype=np.float32))
        with torch.no_grad():
            return np.clip(self.model(x).numpy() * self.cap, 0, None)