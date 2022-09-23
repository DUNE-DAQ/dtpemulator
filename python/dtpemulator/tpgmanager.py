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

    def run_packet(self, rtpc_df, channel, ini_pedestal, ini_accum, ssr_adcs, tov_min) -> list:
        tstamp = rtpc_df.index[0].astype(int)
        adcs = rtpc_df.values
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        pedestal = tpg.pedestal_subtraction(adcs, ini_pedestal, ini_accum)
        adcs_sub = np.array(pedestal[0])
        median = pedestal[1][0]
        ini_pedestal = pedestal[1][-1]
        accumulator = pedestal[2][0]
        ini_accum = pedestal[2][-1]
        fir = tpg.fir_filter(np.concatenate((ssr_adcs, adcs_sub)).astype(int))
        adcs_fir = np.array(fir)[31:]
        hits = np.array(tpg.hit_finder(adcs_fir, tov_min))
        aux = np.ones(len(hits))

        tp_array = np.column_stack((tstamp*aux, channel*aux, median*aux, accumulator*aux, hits)).astype(int)

        return tp_array, adcs_sub, adcs_fir, ini_pedestal, ini_accum


    def run_channel(self, rtpc_df, channel, pedchan=False) -> list:
        if pedchan:
            self.initial_pedestal = rtpc_df[channel].values[0]
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        n_packets = rtpc_df[channel].shape[0]//64
        
        timestamps = rtpc_df[channel].index.astype(int)[:64*n_packets]

        ini_pedestal = self.initial_pedestal
        ini_accum = 0
        ssr_adcs = np.zeros(31)

        tp_array = []
        ped_wave = []
        fir_wave = []

        for i in range(n_packets):
            tp_packet, adcs_sub, adcs_fir, ini_pedestal, ini_accum = self.run_packet(rtpc_df[channel].iloc[i*64:64+i*64], channel, ini_pedestal, ini_accum, ssr_adcs, tov_min=4)
            ssr_adcs = adcs_sub[-31:]
            ped_wave.append(adcs_sub)
            fir_wave.append(adcs_fir)
            if(len(tp_packet) > 0): tp_array.append(tp_packet)

        tp_array = np.concatenate((tp_array))
        tp_df = pd.DataFrame(tp_array, columns=['ts', 'offline_ch', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'sum_adc', 'hit_continue'])

        ped_wave = np.concatenate((ped_wave))
        fir_wave = np.concatenate((fir_wave))

        ped_df = pd.DataFrame(ped_wave, index=timestamps, columns=[str(channel)])
        fir_df = pd.DataFrame(fir_wave, index=timestamps, columns=[str(channel)])

        return tp_df, ped_df, fir_df

