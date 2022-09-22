import numpy as np
import pandas as pd
import collections
import rich
from rich import print

import dtpemulator

class TPGManager:

    def __init__(self, initial_pedestal, fir_path, fir_shift, threshold) -> None:

        self.initial_pedestal = initial_pedestal
        self.fir_path = fir_path
        self.fir_shift = fir_shift
        self.threshold = threshold
    
    def pedestal_subtraction(self, adcs) -> list:
        tpg = dtpemulator.TPGenerator(self.initial_pedestal, self.fir_path, self.fir_shift, self.threshold)
        pedestal = tpg.pedestal_subtraction(adcs)
        return pedestal

    def fir_filter(self, adcs) -> list:
        tpg = dtpemulator.TPGenerator(self.initial_pedestal, self.fir_path, self.fir_shift, self.threshold)
        fir = tpg.fir_filter(adcs)
        return fir

    def hit_finder(self, adcs, tov_min=4) -> list:
        tpg = dtpemulator.TPGenerator(self.initial_pedestal, self.fir_path, self.fir_shift, self.threshold)
        hits = tpg.hit_finder(adcs, tov_min)
        return hits