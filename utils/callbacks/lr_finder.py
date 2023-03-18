from lightning.pytorch.callbacks import LearningRateFinder as _LearningRateFinder


class LearningRateFinder(_LearningRateFinder):
    def lr_find(self, *args, **kwargs) -> None:
        super().lr_find(*args, **kwargs)

        # Plot
        fig = self.optimal_lr.plot(suggest=True)
        fig.savefig("lr_finder.png")
