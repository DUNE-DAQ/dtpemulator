#!/usr/bin/env python

from dtpfeedbacktools.rawdatamanager import RawDataManager
from dtpemulator.tpgmanager import TPGManager
import sys
import os.path
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

hit_labels = {'start_time':("start time", " [ticks]"), 'end_time':("end time", " [ticks]"), 'peak_time':("peak time", " [ticks]"), 'peak_adc':("peak adc", " [ADCs]"), 'sum_adc':("sum adc", " [adc]")}

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

    if os.path.isdir(files_path):
        rdm = RawDataManager(files_path, frame_type, map_id)
        tp_files, adc_files = sorted(rdm.list_files(), reverse=True)
        
        rich.print(adc_files)
        
        rtpc_df = rdm.load_tpcs(adc_files[0])
        rich.print(rtpc_df)
        ts_tpc_min, ts_tpc_max = rdm.find_tpc_ts_minmax(adc_files[0])

        ts_tp_min, ts_tp_max = rdm.find_tp_ts_minmax(tp_files[0])
        overlaps = overlap_boundaries([ts_tp_min, ts_tp_max], [ts_tpc_min, ts_tpc_max])
        offset_low, offset_high = rdm.linear_search_tp(tp_files[0], overlaps[0], overlaps[1])

        rtp_df = rdm.load_tps(tp_files[0], int((offset_high-offset_low)//tp_block_bytes), int(offset_low//tp_block_bytes))
        rich.print(rtp_df)

    else:
        rtp_df = pd.read_hdf(files_path, 'raw_fwtps')
        rtpc_df = pd.read_hdf(files_path, 'raw_adcs')

    # y = lambda x: (ts_tpc_min+32*x-ts_tp_min)%2048
    # seq = np.arange(0,64,1)
    # min_remainder = np.min(list(map(y, seq)))
    # min_indx = np.argmin(list(map(y, seq)))
    # min_shift = seq[min_indx]
    # rich.print(min_remainder, min_shift)

    tpgm = TPGManager(8000, "data/fir_coeffs.dat", 6, 100)
    tp_df, ped_df, fir_df = tpgm.run_capture(rtpc_df, ts_tpc_min, ts_tp_min, pedchan=True)

    rich.print(tp_df)

    tp_df.to_hdf("emu_tp.hdf5", key="tp")

    comp_df = pd.merge(tp_df, rtp_df, on=["ts", "offline_ch"])
    comp_df.to_hdf("comp.hdf5", key="comp")

    out_path = "./plots_2d.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_path)
    for var in hit_labels.keys():
        fig = plt.figure()
        plt.style.use('ggplot')
        ax = fig.add_subplot(111)

        if((var == "peak_adc")or(var == "sum_adc")):
            out_min = min(np.min(comp_df[var+"_x"]), np.min(comp_df[var+"_y"]))
            out_max = max(np.max(comp_df[var+"_x"]), np.max(comp_df[var+"_y"]))
            plt.hist2d(comp_df[var+"_x"], comp_df[var+"_y"], bins=[np.linspace(out_min,out_max,100), np.linspace(out_min,out_max,100)], cmap="plasma", norm=matplotlib.colors.LogNorm())
        else:
            plt.hist2d(comp_df[var+"_x"], comp_df[var+"_y"], bins=[np.arange(0,64,1), np.arange(0,64,1)], cmap="plasma", norm=matplotlib.colors.LogNorm())

        plt.xlabel("Emulated "+hit_labels[var][0]+hit_labels[var][1], fontsize=14, labelpad=10, loc="right")
        plt.ylabel("Firmware "+hit_labels[var][0]+hit_labels[var][1], fontsize=14, labelpad=10, loc="top")

        pdf.savefig()
        plt.close()
    pdf.close()

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
