from tqdm.notebook import tqdm
import psutil
import os

class TqdmProgressBar:
    def __init__(self):
        self.last_file = None

    def run(self, iterable, label, func):
        with tqdm(
                iterable,
                desc=label,
                ncols=777,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"
        ) as progress:
            for item in progress:
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                progress.set_postfix({
                    "Last": " "+str(self.last_file or '–')[:18],
                    "Memory": f" {mem:.1f}MB"
                })
                func(item)
                self.last_file = str(item)
