import numpy as np
import pandas as pd
import collections
import rich
from rich import print
from rich.progress import track

import detchannelmaps
import dtpemulator

class VSTChannelMap(object):

    @staticmethod
    def get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, fiber_no, ch_no):
        return 256*fiber_no+ch_no

    @staticmethod
    def get_plane_from_offline_channel(ch):
        return 0

class TPGManager:

    @staticmethod
    def make_channel_map(map_id):

        if map_id == 'VDColdbox':
            return detchannelmaps.make_map('VDColdboxChannelMap')
        elif map_id == 'HDColdbox':
            return detchannelmaps.make_map('HDColdboxChannelMap')
        elif map_id == 'ProtoDUNESP1':
            return detchannelmaps.make_map('ProtoDUNESP1ChannelMap')
        elif map_id == 'PD2HD':
            return detchannelmaps.make_map('PD2HDChannelMap')
        elif map_id == 'VST':
            return VSTChannelMap()
        else:
            raise RuntimeError(f"Unknown channel map id '{map_id}'")

    def __init__(self, initial_pedestal, fir_path, fir_shift, threshold, ch_map_id="HDColdbox") -> None:

        self.initial_pedestal = initial_pedestal
        self.fir_path = fir_path
        self.fir_shift = fir_shift
        self.threshold = threshold
        self.ch_map = self.make_channel_map(ch_map_id)
    
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

    #def hit_finder(self, rtpc_df, channel, tov_min=4, align=False) -> list:
    def hit_finder(self, adcs, tov_min=4) -> list:
        """
        if align:
            y = lambda x: (ts_tpc_min+32*x-ts_tp)%2048
            seq = np.arange(0,64,1)
            min_remainder = np.min(list(map(y, seq)))
            min_indx = np.argmin(list(map(y, seq)))
            shift = seq[min_indx]
        else:
            shift = 0
        
        n_packets = (rtpc_df[channel].shape[0]-shift)//64 
        timestamps = rtpc_df[channel].index.astype(int)[shift:64*n_packets+shift]

        ini_pedestal = self.initial_pedestal
        ini_accum = 0
        ssr_adcs = np.zeros(31)

        tp_array = []

        for i in range(n_packets):
            adcs = rtpc_df[channel].iloc[shift+i*64:64+shift+i*64].values
            rich.print(adcs)
            tp_packet, adcs_sub, adcs_fir, ini_pedestal, ini_accum = self.run_packet(rtpc_df[channel].iloc[shift+i*64:64+shift+i*64], channel, ini_pedestal, ini_accum, ssr_adcs, tov_min=4, ped_debug=False)
            ssr_adcs = adcs_sub[-31:]
            if(len(tp_packet) > 0): tp_array.append(tp_packet)
        """
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        hits = tpg.hit_finder(adcs, tov_min)
        return hits

    def run_packet(self, rtpc_df, channel, ini_pedestal, ini_accum, ssr_adcs, tov_min, skip_hf) -> list:
        tstamp = rtpc_df.index[0].astype(int)
        #rich.print("in packet", tstamp)
        adcs = rtpc_df.values
        tpg = dtpemulator.TPGenerator(self.fir_path, self.fir_shift, self.threshold)
        pedestal = tpg.pedestal_subtraction(adcs, ini_pedestal, ini_accum)
        adcs_sub = np.array(pedestal[0])
        pedval = np.array(pedestal[1])
        median = pedestal[1][0]
        ini_pedestal = pedestal[1][-1]
        accumulator = pedestal[2][0]
        ini_accum = pedestal[2][-1]
        fir = tpg.fir_filter(np.concatenate((ssr_adcs, adcs_sub)).astype(int))
        adcs_fir = np.array(fir)[31:]
        if skip_hf:
            return adcs_sub, pedval, adcs_fir, ini_pedestal, ini_accum

        hits = np.array(tpg.hit_finder(adcs_fir, tov_min))
        aux = np.ones(len(hits)).astype(int)

        tp_array = np.column_stack((tstamp*aux, channel*aux, median*aux, accumulator*aux, hits)).astype(int)
        #rich.print("in tp array", tp_array[:,0])

        return tp_array, adcs_sub, pedval, adcs_fir, ini_pedestal, ini_accum


    def run_channel(self, rtpc_df, channel, shift=0, pedchan=False, skip_hf=False) -> list:
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
        pedval_list = []
        fir_wave = []

        #rich.print("first ts in list", timestamps[0])

        for i in range(n_packets):
            if skip_hf:
                adcs_sub, pedval, adcs_fir, ini_pedestal, ini_accum = self.run_packet(rtpc_df[channel].iloc[shift+i*64:64+shift+i*64], channel, ini_pedestal, ini_accum, ssr_adcs, tov_min=4, ped_debug=ped_debug)
            else:
                tp_packet, adcs_sub, pedval, adcs_fir, ini_pedestal, ini_accum = self.run_packet(rtpc_df[channel].iloc[shift+i*64:64+shift+i*64], channel, ini_pedestal, ini_accum, ssr_adcs, tov_min=4, ped_debug=ped_debug)
                if(len(tp_packet) > 0): tp_array.append(tp_packet)

            ssr_adcs = adcs_sub[-31:]
            ped_wave.append(adcs_sub)
            pedval_list.append(pedval)
            fir_wave.append(adcs_fir)

        ped_wave = np.concatenate((ped_wave))
        pedval_list = np.concatenate((pedval_list))
        fir_wave = np.concatenate((fir_wave))

        ped_df = pd.DataFrame(ped_wave, index=timestamps, columns=[channel])
        pedval_df = pd.DataFrame(pedval_list, index=timestamps, columns=[channel])
        fir_df = pd.DataFrame(fir_wave, index=timestamps, columns=[channel])

        if skip_hf:
            return ped_df, pedval_df, fir_df

        if(len(tp_array) == 0):
            tp_array = -np.ones((1,10), dtype=int)
            tp_df = pd.DataFrame(tp_array, columns=['ts', 'offline_ch', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'sum_adc', 'hit_continue'])
            return tp_df, ped_df, fir_df

        tp_array = np.concatenate((tp_array))
        tp_df = pd.DataFrame(tp_array, columns=['ts', 'offline_ch', 'median', 'accumulator', 'start_time', 'end_time', 'peak_time', 'peak_adc', 'sum_adc', 'hit_continue'])

        return tp_df, ped_df, pedval_df, fir_df

    def run_capture(self, rtpc_df, ts_tpc_min, ts_tp_min, pedchan=False, align=True, skip_hf=False) -> list:
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
        pedval_df = []
        fir_df = []
        for chan in track(chan_list, description="Processing channels..."):
            if skip_hf:
                ped_chan_df, pedval_chan_df, fir_chan_df = self.run_channel(rtpc_df, chan, shift, pedchan)
            else:
                tp_chan_df, ped_chan_df, pedval_chan_df, fir_chan_df = self.run_channel(rtpc_df, chan, shift, pedchan)
                tp_df.append(tp_chan_df)
            ped_df.append(ped_chan_df)
            pedval_df.append(pedval_chan_df)
            fir_df.append(fir_chan_df)
        
        ped_df = pd.concat(ped_df, axis=1)
        pedval_df = pd.concat(pedval_df, axis=1)
        fir_df = pd.concat(fir_df, axis=1)

        if skip_hf:
            return ped_df, pedval_df, fir_df

        tp_df = pd.concat(tp_df, ignore_index=True)
        tp_df = tp_df.loc[tp_df['ts'] != -1].reset_index(drop=True)

        return tp_df, ped_df, pedval_df, fir_df
