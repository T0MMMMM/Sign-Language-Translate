import pickle
import json
import os
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_PATH = "models/asl_model.pkl"
ONNX_OUT = "web/models/asl_model.onnx"
LABELS_OUT = "web/models/labels.json"


def export():
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : {MODEL_PATH} introuvable. Lance d'abord train_kaggle.py.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Modèle chargé. Classes : {model.classes_}")

    initial_type = [("float_input", FloatTensorType([None, 63]))]
    options = {type(model): {"zipmap": False}}
    onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

    os.makedirs("web/models", exist_ok=True)
    with open(ONNX_OUT, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX sauvegardé : {ONNX_OUT}")

    input_name = onnx_model.graph.input[0].name
    output_names = [o.name for o in onnx_model.graph.output]
    print(f"  Input name  : {input_name}")
    print(f"  Output names: {output_names}")

    labels = model.classes_.tolist()
    with open(LABELS_OUT, "w") as f:
        json.dump(labels, f)
    print(f"Labels sauvegardes : {LABELS_OUT} -> {labels}")


if __name__ == "__main__":
    export()
