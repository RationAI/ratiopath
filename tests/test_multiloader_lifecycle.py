import pytest

from ratiopath.lightning.callbacks import MultiloaderLifecycle


class Recorder(MultiloaderLifecycle):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, int]] = []

    def on_validation_dataloader_start(self, trainer, pl_module, dataloader_idx):
        self.events.append(("validation", "start", dataloader_idx))

    def on_validation_dataloader_end(self, trainer, pl_module, dataloader_idx):
        self.events.append(("validation", "end", dataloader_idx))

    def on_test_dataloader_start(self, trainer, pl_module, dataloader_idx):
        self.events.append(("test", "start", dataloader_idx))

    def on_test_dataloader_end(self, trainer, pl_module, dataloader_idx):
        self.events.append(("test", "end", dataloader_idx))

    def on_predict_dataloader_start(self, trainer, pl_module, dataloader_idx):
        self.events.append(("predict", "start", dataloader_idx))

    def on_predict_dataloader_end(self, trainer, pl_module, dataloader_idx):
        self.events.append(("predict", "end", dataloader_idx))


@pytest.mark.parametrize("stage", ["validation", "test", "predict"])
def test_dataloader_hooks_follow_index_changes(stage):
    callback = Recorder()
    batch_start = getattr(callback, f"on_{stage}_batch_start")
    epoch_end = getattr(callback, f"on_{stage}_epoch_end")

    for batch_idx, dataloader_idx in enumerate([0, 0, 1, 1, 1, 2]):
        batch_start(None, None, None, batch_idx, dataloader_idx=dataloader_idx)
    epoch_end(None, None)

    assert callback.events == [
        (stage, "start", 0),
        (stage, "end", 0),
        (stage, "start", 1),
        (stage, "end", 1),
        (stage, "start", 2),
        (stage, "end", 2),
    ]


def test_state_resets_between_epochs():
    callback = Recorder()

    for _ in range(2):
        callback.on_validation_batch_start(None, None, None, 0, dataloader_idx=0)
        callback.on_validation_epoch_end(None, None)

    assert (
        callback.events
        == [
            ("validation", "start", 0),
            ("validation", "end", 0),
        ]
        * 2
    )


def test_epoch_end_without_batches_calls_nothing():
    callback = Recorder()

    callback.on_test_epoch_end(None, None)

    assert callback.events == []
