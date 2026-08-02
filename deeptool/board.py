"""Live training curves that update in place inside a notebook cell."""

from collections.abc import Sequence

from matplotlib import pyplot as plt

from deeptool.core import HyperParameters


def _in_notebook() -> bool:
    """True when running inside an IPython kernel, False in a plain script."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


class ProgressBoard(HyperParameters):
    """A live plot that accumulates one curve per label.

    Each `draw` call buffers a point. Once `every_n` points have arrived they
    are averaged into a single point on the curve and the figure is redrawn.

    In a notebook the previous output is replaced with the new figure. In a
    plain script the data still accumulates but nothing is rendered.
    """

    def __init__(self, xlabel: str | None = None, ylabel: str | None = None,
                 xlim: tuple[float, float] | None = None,
                 ylim: tuple[float, float] | None = None,
                 xscale: str = "linear", yscale: str = "linear",
                 ls: Sequence[str] = ("-", "--", "-.", ":"),
                 colors: Sequence[str] = ("C0", "C1", "C2", "C3"),
                 figsize: tuple[float, float] = (3.5, 2.5),
                 display: bool = True) -> None:
        self.save_hyperparameters()
        self.raw_points = {}
        self.data = {}
        self.fig = None
        self.axes = None

    def draw(self, x: float, y: float, label: str, every_n: int = 1) -> None:
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        points.append((float(x), float(y)))
        if len(points) < every_n:
            return
        self.data[label].append((
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        ))
        points.clear()
        if self.display:
            self._render()

    def _render(self) -> None:
        if self.fig is None:
            self.fig, self.axes = plt.subplots(figsize=self.figsize)
            # pyplot 의 figure 매니저에서 떼어내 inline 백엔드가 셀 끝에서
            # 같은 그림을 한 번 더 출력하는 것을 막는다. 참조는 우리가 들고 있다.
            plt.close(self.fig)
        self.axes.cla()
        for i, (label, line) in enumerate(self.data.items()):
            self.axes.plot(
                [p[0] for p in line], [p[1] for p in line],
                linestyle=self.ls[i % len(self.ls)],
                color=self.colors[i % len(self.colors)],
                label=label,
            )
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        if self.xlim is not None:
            self.axes.set_xlim(self.xlim)
        if self.ylim is not None:
            self.axes.set_ylim(self.ylim)
        self.axes.grid(True)
        self.axes.legend()
        self._show()

    def _show(self) -> None:
        if not _in_notebook():
            return
        from IPython import display as ipy_display

        ipy_display.clear_output(wait=True)
        ipy_display.display(self.fig)
