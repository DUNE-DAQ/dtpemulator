import numpy as np
import pandas as pd
import collections
import rich
from rich import print
from rich.progress import track

import dtpemulator

class TPGManager:

    def __init__(self, initial_pedestal, fir_path, fir_shift, threshold) -> None:

        self.initial_pedestal = initial_pedestal
        self.fir_path = fir_path
        self.fir_shift = fir_shift
        self.threshold = threshold
    
    def pedestal_subtraction(self, rtpc_df, channel, pedchan=False) -> list:
        if pedchan:
            self.initial_pedestal = rtpc_df[channel].values[0]
        timestamps = rtpc_df[channel].index.astype(int)
        adcs = rtpc_df[channel].values
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        pedsub, pedval, accum = tpg.pedestal_subtraction(adcs, self.initial_pedestal, 0, 10)
        pedsub_df = pd.DataFrame(pedsub, index=timestamps, columns=[channel])
        pedval_df = pd.DataFrame(pedval, index=timestamps, columns=[channel])
        accum_df = pd.DataFrame(accum, index=timestamps, columns=[channel])
        return pedsub_df, pedval_df, accum_df

    def fir_filter(self, rtpc_df, channel) -> list:
        timestamps = rtpc_df[channel].index.astype(int)
        adcs = rtpc_df[channel].values
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        fir = tpg.fir_filter(adcs)
        fir_df = pd.DataFrame(fir, index=timestamps, columns=[channel])
        return fir_df

    def hit_finder(self, adcs, tov_min=4) -> list:
        tpg = dtpemulator.TPGenerator(self.initial_pedestal, self.fir_path, self.fir_shift, self.threshold)
        hits = tpg.hit_finder(adcs, tov_min)
        return hits

    def run_packet(self, rtpc_df, channel, ini_pedestal, ini_accum, ssr_adcs, tov_min) -> list:
        tstamp = rtpc_df.index[0].astype(int)
        #rich.print("in packet", tstamp)
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
        aux = np.ones(len(hits)).astype(int)

        tp_array = np.column_stack((tstamp*aux, channel*aux, median*aux, accumulator*aux, hits)).astype(int)
        #rich.print("in tp array", tp_array[:,0])

        return tp_array, adcs_sub, adcs_fir, ini_pedestal, ini_accum


    def run_channel(self, rtpc_df, channel, shift=0, pedchan=False) -> list:
        if pedchan:
            self.initial_pedestal = rtpc_df[channel].values[0]
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        n_packets = (rtpc_df[channel].shape[0]-shift)//64
        
        timestamps = rtpc_df[channel].index.astype(int)[shift:64*n_packets+shift]

        ini_pedestal = self.initial_pedestal
        ini_accum = 0
        ssr_adcs = np.zeros(31)

        tp_array = []
        ped_wave = []
        fir_wave = []

        #rich.print("first ts in list", timestamps[0])

        for i in range(n_packets):
            
            #rich.print(rtpc_df[channel].iloc[shift+i*64:64+shift+i*64].values, channel)
            tp_packet, adcs_sub, adcs_fir, ini_pedestal, ini_accum = self.run_packet(rtpc_df[channel].iloc[shift+i*64:64+shift+i*64], channel, ini_pedestal, ini_accum, ssr_adcs, tov_min=4)
            ssr_adcs = adcs_sub[-31:]
            ped_wave.append(adcs_sub)
            fir_wave.append(adcs_fir)
            if(len(tp_packet) > 0): tp_array.append(tp_packet)

        ped_wave = np.concatenate((ped_wave))
        fir_wave = np.concatenate((fir_wave))

        ped_df = pd.DataFrame(ped_wave, index=timestamps, columns=[str(channel)])
        fir_df = pd.DataFrame(fir_wave, index=timestamps, columns=[str(channel)])

        if(len(tp_array) == 0):
            tp_array = -np.ones((1,10))
            tp_df = pd.DataFrame(tp_array, columns=['ts', 'offline_ch', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'sum_adc', 'hit_continue'])
            return tp_df, ped_df, fir_df

        tp_array = np.concatenate((tp_array))
        tp_df = pd.DataFrame(tp_array, columns=['ts', 'offline_ch', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'sum_adc', 'hit_continue'])

        return tp_df, ped_df, fir_df

    def run_capture(self, rtpc_df, ts_tpc_min, ts_tp_min, pedchan=False, align=True) -> list:
        if align:
            y = lambda x: (ts_tpc_min+32*x-ts_tp_min)%2048
            seq = np.arange(0,64,1)
            min_remainder = np.min(list(map(y, seq)))
            min_indx = np.argmin(list(map(y, seq)))
            shift = seq[min_indx]
        else:
            shift = 0

        chan_list = rtpc_df.keys()
        tp_df = []
        ped_df = []
        fir_df = []
        for chan in track(chan_list, description="Processing channels..."):
            tp_chan_df, ped_chan_df, fir_chan_df = self.run_channel(rtpc_df, chan, shift, pedchan)
            tp_df.append(tp_chan_df)
            ped_df.append(ped_chan_df)
            fir_df.append(fir_chan_df)
        
        tp_df = pd.concat(tp_df, ignore_index=True)
        ped_df = pd.concat(ped_df, axis=1)
        fir_df = pd.concat(fir_df, axis=1)

        return tp_df, ped_df, fir_df
