import os, sys, types, torch, pickle
from train_iterative import LSTMAttn

class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "LSTMAttn" and module in ("__main__", "train_iterative"):
            return LSTMAttn
        return super().find_class(module, name)

_CustomPickle = types.ModuleType("pickle")
_CustomPickle.Unpickler = _RenameUnpickler

def main():
    here = os.path.dirname(__file__)
    src = os.path.join(here, "best_model.pth")
    dst = os.path.join(here, "best_model_v2.pth")

    if not os.path.exists(src):
        print(f"Missing source model: {src}")
        sys.exit(1)

    model = torch.load(src, map_location="cpu", pickle_module=_CustomPickle)
    torch.save(model, dst)
    print(f"Re-saved model to: {dst}")

if __name__ == "__main__":
    main()