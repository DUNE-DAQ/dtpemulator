#!/usr/bin/env python

from dtpfeedbacktools.rawdatamanager import RawDataManager
from dtpemulator.tpgmanager import TPGManager
import sys
import rich
from rich.table import Table
import logging
import click
from rich import print
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

tp_block_size = 3
tp_block_bytes = tp_block_size*4

NS_PER_TICK = 16

def signal_to_noise(array):
    sigma_raw = np.std(array)
    noise_array = array[(-3*sigma_raw <= array)&(array <= +3*sigma_raw)]
    mu_noise = np.mean(noise_array)
    sigma_noise = np.std(noise_array)
    return (np.max(array)-mu_noise)/sigma_noise

def overlap_check(tp_tstamp, adc_tstamp):
    overlap_true = (adc_tstamp[0] <= tp_tstamp[1])&(adc_tstamp[1] >= tp_tstamp[0])
    overlap_time = max(0, min(tp_tstamp[1], adc_tstamp[1]) - max(tp_tstamp[0], adc_tstamp[0]))
    return np.array([overlap_true, overlap_time])

def overlap_boundaries(tp_tstamp, adc_tstamp):
    return [max(tp_tstamp[0], adc_tstamp[0]), min(tp_tstamp[1], adc_tstamp[1])]

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('-t', '--threshold', help="Threshold to be used in emulation", default=20)
@click.option('-o', '--out_file', help="Output file name", default=".")

def cli(interactive: bool, files_path: str, threshold: int = 20, out_file: str = ".") -> None:

    rtpc_df = pd.read_hdf(files_path, 'rtpc')
    rich.print(rtpc_df)

    tpgm = TPGManager(500, "data/fir_coeffs.dat", 6, threshold)
    #ped_df, pedval_df, fir_df = tpgm.run_capture(rtpc_df, 0, 0, pedchan=True, align=False, skip_hf=True)
    pedsub_df, pedval_df, accum_df = tpgm.pedestal_subtraction(rtpc_df, pedchan=True)
    fir_df = tpgm.fir_filter(pedsub_df)

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
