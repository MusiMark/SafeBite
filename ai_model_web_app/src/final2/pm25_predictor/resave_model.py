import os, pickle, sys
from train_model import (
    OptimizedPM25Predictor, OptimizedGNNPredictor, OptimizedKGBuilder
)

class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            if name == "OptimizedPM25Predictor":
                return OptimizedPM25Predictor
            if name == "OptimizedGNNPredictor":
                return OptimizedGNNPredictor
            if name == "OptimizedKGBuilder":
                return OptimizedKGBuilder
        return super().find_class(module, name)

def main():
    here = os.path.dirname(__file__)
    src = os.path.join(here, "pm25_gnn_complete_complete_model.pkl")
    dst = os.path.join(here, "pm25_gnn_complete_complete_model_v2.pkl")

    if not os.path.exists(src):
        print(f"Missing source model: {src}")
        sys.exit(1)

    with open(src, "rb") as f:
        predictor = _RenameUnpickler(f).load()

    with open(dst, "wb") as f:
        pickle.dump(predictor, f)

    print(f"Re-saved model to: {dst}")

if __name__ == "__main__":
    main()