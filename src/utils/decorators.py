def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        prev_state = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(prev_state)

        return out

    return inner
