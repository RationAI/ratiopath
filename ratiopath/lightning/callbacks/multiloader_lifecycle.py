from abc import ABC
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class MultiloaderLifecycle(Callback, ABC):
    """Callback base class with hooks for the start and end of each dataloader.

    With several validation, test or predict dataloaders (e.g. one per slide), Lightning
    reports only batches and epochs. This class adds `on_{stage}_dataloader_start` and
    `on_{stage}_dataloader_end` hooks, where `stage` is `validation`, `test` or
    `predict`. They are called whenever the dataloader index changes and once more for
    the last dataloader at the end of the epoch.

    Requires the `lightning` extra: `pip install ratiopath[lightning]`.

    Note:
        Subclasses that override `on_{stage}_batch_start` or `on_{stage}_epoch_end`
        must call the parent implementation, otherwise the dataloader hooks are not
        called.

    Examples:
        ```python
        from lightning import LightningModule, Trainer

        from ratiopath.lightning.callbacks import MultiloaderLifecycle


        class SlideHeatmaps(MultiloaderLifecycle):
            def on_predict_dataloader_start(
                self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
            ) -> None: ...  # e.g. allocate a heatmap for the slide

            def on_predict_dataloader_end(
                self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
            ) -> None: ...  # e.g. save the finished heatmap
        ```
    """

    def __init__(self) -> None:
        self._dataloader_idxs = {"validation": -1, "test": -1, "predict": -1}

    def on_validation_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called before the first validation batch of a dataloader."""

    def on_validation_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called after the last validation batch of a dataloader."""

    def on_test_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called before the first test batch of a dataloader."""

    def on_test_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called after the last test batch of a dataloader."""

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called before the first predict batch of a dataloader."""

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        """Called after the last predict batch of a dataloader."""

    def on_batch_start(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx != self._dataloader_idxs[stage]:
            if self._dataloader_idxs[stage] != -1:
                getattr(self, f"on_{stage}_dataloader_end")(
                    trainer, pl_module, self._dataloader_idxs[stage]
                )

            self._dataloader_idxs[stage] = dataloader_idx
            getattr(self, f"on_{stage}_dataloader_start")(
                trainer, pl_module, dataloader_idx
            )

    def on_epoch_end(
        self, stage: str, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self._dataloader_idxs[stage] != -1:
            getattr(self, f"on_{stage}_dataloader_end")(
                trainer, pl_module, self._dataloader_idxs[stage]
            )
            self._dataloader_idxs[stage] = -1

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("validation", trainer, pl_module, dataloader_idx)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_epoch_end("validation", trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("test", trainer, pl_module, dataloader_idx)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_end("test", trainer, pl_module)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("predict", trainer, pl_module, dataloader_idx)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_epoch_end("predict", trainer, pl_module)
