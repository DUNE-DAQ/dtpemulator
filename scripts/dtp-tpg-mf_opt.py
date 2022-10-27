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

def bipolar(x, A, delta, sigma):
    return -2*A*(delta+x)*np.exp(-np.power(x,2)/np.power(sigma,2))

def signal_to_noise(array):
    sigma_raw = np.std(array)
    noise_array = array[(-3*sigma_raw <= array)&(array <= +3*sigma_raw)]
    mu_noise = np.mean(noise_array)
    sigma_noise = np.std(noise_array)
    return (np.max(array)-mu_noise)/sigma_noise

def get_max(df, plane):
    sigma_max, delta_max = df.stack().index[np.argmax(df.values)]
    max_value = df.loc[sigma_max, delta_max]
    rich.print(f'Plane {plane} optimal values: \ndelta = {delta_max}, sigma = {sigma_max} \nsignal-to-noise improvement = {max_value}')
    return sigma_max, delta_max, max_value

def plot_heatmaps(IR_U_df, IR_V_df, IR_X_df):
    fig = plt.figure(figsize=(8,16))

    gs = fig.add_gridspec(3, 1, hspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)

    sns.heatmap(IR_U_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[0])
    sns.heatmap(IR_V_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[1])
    sns.heatmap(IR_X_df.iloc[::-1], center=1, annot=True, cmap="coolwarm", ax=axs[2])

    axs[2].set_xlabel(r"$\delta$")
    axs[0].set_ylabel(r"$\sigma$")
    axs[1].set_ylabel(r"$\sigma$")
    axs[2].set_ylabel(r"$\sigma$")

    plt.savefig("./example_seaborn.png")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--interactive', is_flag=True, default=False)
@click.argument('files_path', type=click.Path(exists=True))
@click.option('--delta_min', help="Minimum value of delta", default=-0.1)
@click.option('--delta_max', help="Maximum value of delta", default=0.1)
@click.option('--delta_len', help="Number of delta values to sample", default=10)
@click.option('--sigma_min', help="Minimum value of sigma", default=0.05)
@click.option('--sigma_max', help="Maximum value of sigma", default=0.5)
@click.option('--sigma_len', help="Number of sigma values to sample", default=10)
@click.option('-o', '--out_file', help="Output file name", default=".")

def cli(interactive: bool, files_path: str, delta_min: int = -0.1, delta_max: int = 0.1, delta_len: int = 10, sigma_min: int = 0.05, sigma_max: int = 0.5, sigma_len: int = 10, out_file: str = ".") -> None:

    rtpc_df = pd.read_hdf(files_path, 'rtpc')
    rich.print(rtpc_df)

    tpgm = TPGManager(500, "data/fir_coeffs.dat", 8, 20)
    pedsub_df, pedval_df, accum_df = tpgm.pedestal_subtraction(rtpc_df, pedchan=True)
    SN_raw = pedsub_df.apply(signal_to_noise, axis=0, raw=True)

    x_range = np.linspace(-0.5,0.5,32)
    amplitude = 1000
    delta_range = np.linspace(delta_min, delta_max, delta_len)
    sigma_range = np.linspace(sigma_min, sigma_max, sigma_len)

    IR_U = []
    IR_V = []
    IR_X = []

    for delta in track(delta_range, description="Applying filters..."):
        IR_U_aux = []
        IR_V_aux = []
        IR_X_aux = []
        for sigma in sigma_range:
            y = [bipolar(k, amplitude, delta, sigma) for k in x_range]
            roots = [(-delta-np.sqrt(np.power(delta,2)+2*np.power(sigma,2)))/2, (-delta+np.sqrt(np.power(delta,2)+2*np.power(sigma,2)))/2]
            y = np.array(y)/np.abs(roots[0])
            taps = y[::-1]

            np.savetxt("data/temp_taps.txt", taps)
            tpgm = TPGManager(500, "data/temp_taps.txt", 8, 20)
            fir_df = tpgm.fir_filter(pedsub_df)
            SN = fir_df.apply(signal_to_noise, axis=0, raw=True)
            IR = SN/SN_raw
            IR_U_aux.append(IR.iloc[IR.index%2560<800].mean())
            IR_V_aux.append(IR.iloc[(IR.index%2560>=800)&(IR.index%2560<1600)].mean())
            IR_X_aux.append(IR.iloc[IR.index%2560>=1600].mean())
        IR_U.append(IR_U_aux)
        IR_V.append(IR_V_aux)
        IR_X.append(IR_X_aux)

    IR_U_df = pd.DataFrame(collections.OrderedDict([('sigma', sigma_range)]+[(delta_range[i], IR_U[i]) for i in range(delta_len)]))
    IR_U_df = IR_U_df.set_index('sigma')
    IR_V_df = pd.DataFrame(collections.OrderedDict([('sigma', sigma_range)]+[(delta_range[i], IR_V[i]) for i in range(delta_len)]))
    IR_V_df = IR_V_df.set_index('sigma')
    IR_X_df = pd.DataFrame(collections.OrderedDict([('sigma', sigma_range)]+[(delta_range[i], IR_X[i]) for i in range(delta_len)]))
    IR_X_df = IR_X_df.set_index('sigma')

    plot_heatmaps(IR_U_df, IR_V_df, IR_X_df)

    get_max(IR_U_df, "U")

    if interactive:
        import IPython
        IPython.embed()

if __name__ == "__main__":

    cli()
