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

    """
    rdm = RawDataManager(files_path, frame_type, map_id)
    tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
    
    rich.print(adc_files)
    
    rtpc_df = rdm.load_tpcs(adc_files[0])
    rich.print(rtpc_df)
    """

    rtpc_df = pd.read_hdf(files_path, 'rtpc')
    rich.print(rtpc_df)

    tpgm = TPGManager(500, "data/fir_coeffs.dat", 6, 20)
    tp_df, ped_df, fir_df = tpgm.run_capture(rtpc_df, 0, 0, pedchan=False, align=False)

    rich.print(tp_df)
    tp_df.to_hdf("UniqueHits_F_emulator_rtp.hdf5", "emulator_rtp")
    rich.print(ped_df)
    #tp_df.to_hdf("../UniqueHits_F_emulator_rtp.hdf5", "emulator_rtp")
    rich.print(fir_df)
    #fir_df.to_hdf("UniqueHits_F_emulator_fir.hdf5", "emulator_fir")

    return

    for i in range(4):
        plt.axvline(x=i*64, linestyle=":", c="black", alpha=0.5)
    plt.plot(ped_df.values)
    plt.plot(fir_df.values, marker="x")
    plt.scatter((tp_df["ts"]-tp_df["ts"].min())//25+tp_df["peak_time"], tp_df["peak_adc"], marker="v", c="r")
    plt.axhline(y=20, c="g", linestyle="--", alpha=0.7)
    plt.savefig("test.png")

    return

    out_path = "./plots.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
    for j in range(len(tp_df)):

        tstamp = tp_df["ts"][j]
        start_time = tp_df["start_time"][j]
        end_time = tp_df["end_time"][j]
        peak_time = tp_df["peak_time"][j]
        fw_median = tp_df["median"][j]

        fig = plt.figure()
        plt.style.use('ggplot')
        ax = fig.add_subplot(111)

        #plt.plot(rtpc_df[199].values, c="dodgerblue", label="Raw")
        plt.plot(ped_df, c="red", alpha=0.8, label="PedSub")
        plt.plot(fir_df, c="green", alpha=0.6, label="FIR")

        for i in range(4):
            plt.axvline(x=-2048+i*2048+tstamp, linestyle=":", c="k", alpha=0.2)
    
        ax.axvspan(start_time*32+tstamp, end_time*32+tstamp, alpha=0.4, color='red')
        ax.axvline(x=peak_time*32+tstamp, linestyle="--", alpha=0.6, color='black')

        #ax.hlines(y=fw_median, xmin=tstamp, xmax=2048+tstamp, linestyle="-.", colors="black", alpha=0.5, label="median")
        ax.hlines(y=100, xmin=tstamp, xmax=2048+tstamp, linestyle="-.", colors="limegreen", alpha=0.5, label="threshold")

        plt.xlim(tstamp-2048, tstamp+2*2048)
        plt.ylim(-250, 250)

        plt.xlabel("Time [tstamp]", fontsize=14, labelpad=10, loc="right")
        plt.ylabel("Amplitude [ADC]", fontsize=14, labelpad=10, loc="top")

        legend = plt.legend(fontsize=12, loc="upper right")
        frame = legend.get_frame()
        frame.set_color('white')
        frame.set_alpha(0.8)
        frame.set_linewidth(0)

        pdf.savefig()
        plt.close()

    pdf.close()


    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
