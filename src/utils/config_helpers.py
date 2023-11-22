import importlib


def get_obj_from_str(input: str, reload: bool = False):
    module, cls = input.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_cfg(cfg):
    if "target" not in cfg:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(cfg["target"])(**cfg.get("params", dict()))
