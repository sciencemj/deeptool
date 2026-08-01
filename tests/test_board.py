from ood.board import ProgressBoard


def test_draw_buffers_until_every_n_reached():
    board = ProgressBoard(display=False)
    board.draw(1.0, 10.0, "loss", every_n=2)

    assert board.data["loss"] == []
    assert len(board.raw_points["loss"]) == 1


def test_draw_averages_every_n_points():
    board = ProgressBoard(display=False)
    board.draw(1.0, 10.0, "loss", every_n=2)
    board.draw(3.0, 20.0, "loss", every_n=2)

    assert board.data["loss"] == [(2.0, 15.0)]
    assert board.raw_points["loss"] == []


def test_draw_with_every_n_one_appends_immediately():
    board = ProgressBoard(display=False)
    board.draw(1.0, 10.0, "loss", every_n=1)

    assert board.data["loss"] == [(1.0, 10.0)]


def test_draw_tracks_multiple_labels_independently():
    board = ProgressBoard(display=False)
    board.draw(1.0, 10.0, "train_loss")
    board.draw(1.0, 20.0, "val_loss")

    assert board.data["train_loss"] == [(1.0, 10.0)]
    assert board.data["val_loss"] == [(1.0, 20.0)]


def test_draw_renders_a_figure_when_display_enabled():
    board = ProgressBoard(display=True)
    board.draw(1.0, 10.0, "loss")

    assert board.fig is not None
    assert len(board.axes.get_lines()) == 1


def test_render_reuses_the_same_figure_across_draws():
    board = ProgressBoard(display=True)
    board.draw(1.0, 10.0, "loss")
    first = board.fig
    board.draw(2.0, 9.0, "loss")

    assert board.fig is first
    assert len(board.axes.get_lines()) == 1


def test_display_false_never_creates_a_figure():
    board = ProgressBoard(display=False)
    board.draw(1.0, 10.0, "loss")

    assert board.fig is None


def test_more_labels_than_styles_wraps_around():
    board = ProgressBoard(display=True, ls=("-",), colors=("C0",))
    for i in range(3):
        board.draw(1.0, float(i), f"series{i}")

    assert len(board.axes.get_lines()) == 3
