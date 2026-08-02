"""학습 곡선을 노트북 셀 안에서 제자리 갱신하는 보드."""

from matplotlib import pyplot as plt

from deeptool.core import HyperParameters


def _in_notebook():
    """IPython 커널 안에서 돌고 있으면 True. 순수 스크립트면 False."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


class ProgressBoard(HyperParameters):
    """label 별 곡선을 누적해 그리는 라이브 플롯.

    ``draw`` 를 부를 때마다 점을 버퍼에 넣고, ``every_n`` 개가 모이면
    평균 1점으로 압축해 곡선에 추가한 뒤 figure 를 다시 그린다.
    노트북이면 이전 출력을 지우고 새 figure 로 교체하고, 스크립트면
    데이터만 누적하고 렌더링은 건너뛴다.
    """

    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale="linear", yscale="linear",
                 ls=("-", "--", "-.", ":"),
                 colors=("C0", "C1", "C2", "C3"),
                 figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
        self.raw_points = {}
        self.data = {}
        self.fig = None
        self.axes = None

    def draw(self, x, y, label, every_n=1):
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

    def _render(self):
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

    def _show(self):
        if not _in_notebook():
            return
        from IPython import display as ipy_display

        ipy_display.clear_output(wait=True)
        ipy_display.display(self.fig)
