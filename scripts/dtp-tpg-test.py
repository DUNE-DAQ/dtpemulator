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
@click.option('-f', '--frame_type', type=click.Choice(
    [
        "ProtoWIB",
        "WIB"
    ]),
              help="Select input frame type", default='WIB')
@click.option('-m', '--map_id', type=click.Choice(
    [
        "VDColdbox",
        "HDColdbox",
        "ProtoDUNESP1",
        "PD2HD",
        "VST"
    ]),
              help="Select input channel map", default=None)

def cli(interactive: bool, files_path: str, frame_type: str = 'WIB', map_id: str = "HDColdbox") -> None:

    rdm = RawDataManager(files_path, frame_type, map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(adc_files)
    
    for i in range(9):
        try:
            rtpc_df = rdm.load_tpcs(adc_files[i])
            rich.print(rtpc_df)

            tpgm = TPGManager(800, "data/fir_coeffs.dat", 6, 100)
            pedestal = tpgm.pedestal_subtraction(rtpc_df[2546].values)
            fir = tpgm.fir_filter(pedestal[0])
            break
        except:
            pass
    #hits = tpgm.hit_finder(fir)

    #rich.print(hits[0])
    #rich.print(hits[1])
    #rich.print(hits[2])

    fig = plt.figure()
    plt.style.use('ggplot')
    ax = fig.add_subplot(111)

    #plt.plot(rtpc_df[199].values, c="dodgerblue", label="Raw")
    plt.plot(pedestal[0], c="red", alpha=0.8, label="Pedestal")
    plt.plot(fir, c="green", alpha=0.6, marker="x", label="FIR")
    #for i in range(len(hits[1])):
    #    ax.axvspan(hits[0][i], hits[1][i], alpha=0.4, color='red')
    #    ax.axvline(x=hits[4][i], linestyle="--", alpha=0.6, color='black')

    #plt.xlim(13150, 13300)
    #plt.ylim(-200, 200)
    plt.xlabel("Time [samples]", fontsize=14, labelpad=10, loc="right")
    plt.ylabel("Amplitude [ADC]", fontsize=14, labelpad=10, loc="top")

    legend = plt.legend(fontsize=12, loc="upper right")
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(0.8)
    frame.set_linewidth(0)

    plt.savefig("fir_new.png", dpi=500)

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
