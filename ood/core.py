"""노트북 친화적인 최소 유틸 — 클래스 확장과 하이퍼파라미터 저장."""

import inspect


def add_to_class(Class):
    """데코레이트한 함수를 ``Class`` 의 메서드로 등록한다.

    노트북에서 클래스를 한 셀에 정의하고, 이후 셀에서 메서드를 덧붙일 때 쓴다::

        @add_to_class(MyNet)
        def loss(self, y_hat, y):
            return F.cross_entropy(y_hat, y)

    d2l 원본과 달리 데코레이터가 원래 함수를 그대로 반환한다.
    (원본은 ``None`` 을 반환해 정의된 셀에서 그 이름이 ``None`` 으로 덮인다.)
    """

    def wrapper(func):
        setattr(Class, func.__name__, func)
        return func

    return wrapper


class HyperParameters:
    """``__init__`` 인자를 속성 + ``self.hparams`` 딕셔너리로 저장하는 믹스인."""

    def save_hyperparameters(self, ignore=()):
        """호출한 ``__init__`` 의 인자를 인스턴스에 저장한다.

        ``self`` 와 ``ignore`` 에 든 이름은 제외한다. 지역 변수는 잡지 않고
        호출 프레임의 선언된 인자(위치 인자 + 키워드 전용 인자)만 읽는다.

        서브클래스가 ``super().__init__()`` 을 부르면 각 클래스의 인자가 각각
        저장되지만, ``self.hparams`` 는 마지막 호출의 것으로 덮인다.
        관례상 ``super().__init__()`` 을 먼저 부르고 그 다음에 호출한다.
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
