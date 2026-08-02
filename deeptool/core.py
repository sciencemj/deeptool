"""Notebook-friendly primitives: class extension and hyperparameter capture."""

import inspect


def add_to_class(Class):
    """Register the decorated function as a method on `Class`.

    Lets you define a class in one notebook cell and attach methods from later
    cells, without re-running the class definition.

    Args:
        Class: The class to attach the method to.

    Returns:
        A decorator that registers the function and returns it unchanged, so the
        name stays usable in the cell that defined it.

    Example:
        ```python
        @add_to_class(MyNet)
        def loss(self, y_hat, y):
            return F.cross_entropy(y_hat, y)
        ```
    """

    def wrapper(func):
        setattr(Class, func.__name__, func)
        return func

    return wrapper


class HyperParameters:
    """Mixin that saves `__init__` arguments as attributes and in `self.hparams`."""

    def save_hyperparameters(self, ignore=()):
        """Store the calling `__init__`'s arguments on the instance.

        Reads the declared arguments of the calling frame — positional and
        keyword-only — so local variables are never captured.

        Call `super().__init__()` first and `save_hyperparameters()` second.
        Each class stores its own arguments, but `self.hparams` holds the last
        call's, so the subclass has to run last to win.

        Args:
            ignore: Argument names to leave out of both the attributes and
                `self.hparams`.
        """
        frame = inspect.currentframe().f_back
        code = frame.f_code
        n_args = code.co_argcount + code.co_kwonlyargcount
        names = code.co_varnames[:n_args]
        local_vars = frame.f_locals

        self.hparams = {
            name: local_vars[name]
            for name in names
            if name != "self" and name not in ignore and name in local_vars
        }
        for name, value in self.hparams.items():
            setattr(self, name, value)
