#!/usr/bin/env python

from dtpfeedbacktools.rawdatamanager import RawDataManager
from dtpemulator.tpgmanager import TPGManager
import sys
import rich
from rich.table import Table
import logging
import click
from rich import print
from rich.progress import track
from pathlib import Path

import numpy as np
import pandas as pd
import collections
import seaborn as sns
import scipy.signal
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

def plot_heatmaps(IR_U_df, IR_V_df, IR_X_df):
    fig = plt.figure(figsize=(8,16))

    gs = fig.add_gridspec(3, 1, hspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)

    sns.heatmap(IR_U_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[0])
    sns.heatmap(IR_V_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[1])
    sns.heatmap(IR_X_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[2])

    axs[2].set_xlabel(r"Cutoff frequency [ticks$^{-1}$]")
    axs[0].set_ylabel(r"Transition width [ticks$^{-1}$]")
    axs[1].set_ylabel(r"Transition width [ticks$^{-1}$]")
    axs[2].set_ylabel(r"Transition width [ticks$^{-1}$]")

    plt.savefig("./example_seaborn.png")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('--cutoff_min', help="Minimum value of frequency cutoff", default=0)
@click.option('--cutoff_max', help="Maximum value of frequency cutoff", default=0.2)
@click.option('--cutoff_len', help="Number of frequency cutoff values to sample", default=10)
@click.option('--transwidth_min', help="Minimum value of frequency transition width", default=0.01)
@click.option('--transwidth_max', help="Maximum value of frequency transition width", default=0.1)
@click.option('--transwidth_len', help="Number of frequency transition width values to sample", default=10)
@click.option('-o', '--out_path', help="Output path", default=".")

def cli(interactive: bool, files_path: str, cutoff_min: int = 0, cutoff_max: int = 0.2, cutoff_len: int = 10, transwidth_min: int = 0.01, transwidth_max: int = 0.1, transwidth_len: int = 10, out_path: str = ".") -> None:

    rtpc_df = pd.read_hdf(files_path, 'rtpc')
    rich.print(rtpc_df)

    tpgm = TPGManager(500, "data/fir_coeffs.dat", 6, 20)
    pedsub_df, pedval_df, accum_df = tpgm.pedestal_subtraction(rtpc_df, pedchan=True)
    SN_raw = pedsub_df.apply(signal_to_noise, axis=0, raw=True)

    cutoffs = np.linspace(cutoff_min, cutoff_max, cutoff_len)
    trans_widths = np.linspace(transwidth_min, transwidth_max, transwidth_len)

    IR_U = []
    IR_V = []
    IR_X = []

    for cutoff in track(cutoffs, description="Applying filters..."):
        IR_U_aux = []
        IR_V_aux = []
        IR_X_aux = []
        for trans_width in trans_widths:
            taps = scipy.signal.remez(32,[0, cutoff, cutoff+trans_width, 0.5],[150,0])
            np.savetxt("data/temp_taps.txt", taps)
            tpgm = TPGManager(500, "data/temp_taps.txt", 6, 20)
            fir_df = tpgm.fir_filter(pedsub_df)
            SN = fir_df.apply(signal_to_noise, axis=0, raw=True)
            IR = SN/SN_raw
            IR_U_aux.append(IR.iloc[IR.index%2560<800].mean())
            IR_V_aux.append(IR.iloc[(IR.index%2560>=800)&(IR.index%2560<1600)].mean())
            IR_X_aux.append(IR.iloc[IR.index%2560>=1600].mean())
        IR_U.append(IR_U_aux)
        IR_V.append(IR_V_aux)
        IR_X.append(IR_X_aux)

    IR_U_df = pd.DataFrame(collections.OrderedDict([('trans width', trans_widths)]+[(cutoffs[i], IR_U[i]) for i in range(cutoff_len)]))
    IR_U_df = IR_U_df.set_index('trans width')
    IR_V_df = pd.DataFrame(collections.OrderedDict([('trans width', trans_widths)]+[(cutoffs[i], IR_V[i]) for i in range(cutoff_len)]))
    IR_V_df = IR_V_df.set_index('trans width')
    IR_X_df = pd.DataFrame(collections.OrderedDict([('trans width', trans_widths)]+[(cutoffs[i], IR_X[i]) for i in range(cutoff_len)]))
    IR_X_df = IR_X_df.set_index('trans width')

    plot_heatmaps(IR_U_df, IR_V_df, IR_X_df)

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
