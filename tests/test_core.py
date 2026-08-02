from deeptool.core import HyperParameters, add_to_class


def test_add_to_class_attaches_method():
    class A:
        pass

    @add_to_class(A)
    def greet(self, name):
        return f"hi {name}"

    assert A().greet("bob") == "hi bob"


def test_add_to_class_returns_the_function():
    """d2l 구현과 다른 지점: 데코레이터가 원래 함수를 그대로 돌려준다."""

    class A:
        pass

    @add_to_class(A)
    def greet(self):
        return "hi"

    assert callable(greet)
    assert greet.__name__ == "greet"


def test_add_to_class_overwrites_existing_method():
    class A:
        def greet(self):
            return "old"

    @add_to_class(A)
    def greet(self):
        return "new"

    assert A().greet() == "new"


def test_save_hyperparameters_sets_attributes():
    class Cfg(HyperParameters):
        def __init__(self, lr=0.1, batch_size=32):
            self.save_hyperparameters()

    cfg = Cfg(lr=0.5)
    assert cfg.lr == 0.5
    assert cfg.batch_size == 32


def test_save_hyperparameters_populates_hparams_dict():
    class Cfg(HyperParameters):
        def __init__(self, lr=0.1, batch_size=32):
            self.save_hyperparameters()

    assert Cfg(lr=0.5).hparams == {"lr": 0.5, "batch_size": 32}


def test_save_hyperparameters_respects_ignore():
    class Cfg(HyperParameters):
        def __init__(self, lr=0.1, secret="x"):
            self.save_hyperparameters(ignore=["secret"])

    cfg = Cfg()
    assert cfg.hparams == {"lr": 0.1}
    assert not hasattr(cfg, "secret")


def test_save_hyperparameters_includes_keyword_only_args():
    class Cfg(HyperParameters):
        def __init__(self, lr=0.1, *, seed=7):
            self.save_hyperparameters()

    assert Cfg().hparams == {"lr": 0.1, "seed": 7}


def test_save_hyperparameters_uses_calling_frame_not_subclass_init():
    """부모의 __init__ 에서 호출하면 부모의 인자만 잡아야 한다."""

    class Base(HyperParameters):
        def __init__(self, root="/data"):
            self.save_hyperparameters()

    class Child(Base):
        def __init__(self, lr=0.1):
            super().__init__()
            self.save_hyperparameters()

    child = Child()
    assert child.root == "/data"
    assert child.lr == 0.1
    assert child.hparams == {"lr": 0.1}


def test_save_hyperparameters_ignores_local_variables():
    class Cfg(HyperParameters):
        def __init__(self, lr=0.1):
            scratch = lr * 2
            self.save_hyperparameters()
            assert scratch == 0.2

    assert Cfg().hparams == {"lr": 0.1}
