# matching mfhTime
import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import click
from tqdm import tqdm
import glob





def checkTabletop(df):
    df = df[df.type == 'tabletop']
    diff = np.diff(df.mfhTime.values)
    #mask = diff != 0.01
    #tuned for 1000 Hz
    mask =  (diff >= 0.0019*1e15)  & (diff <= 0.0021*1e15)

    df['validRef'] = [True]*len(df.index)
    df.validRef[1:] = mask
    print(f'Number of reference triggers: {len(df.index)}')
    print(f'Number of valid reference triggers: {np.sum(mask)}')
    
    return df

def compareTimes(df_total, df_ref,rate=500,spe=True):
    df_ref = df_ref[df_ref.validRef == True]
    df_degg = df_total[df_total.type == 'degg']
    
    ref_time = df_ref.mfhTime.values/1e15 + df_ref.delta.values[0]
    min_time = np.min(ref_time)
    
    for port in df_degg.port.unique():
        for channel in [1]:
            _df = df_degg[(df_degg.port == port) & (df_degg.channel == channel)]
            mfhTime = _df.mfhTime.values/1e15 + _df.delta.values[0] - min_time
            if len(mfhTime) == 0:
                continue
                
    diff = np.diff(df_degg.mfhTime.values)
    diff = np.append(diff, 1e25)
    df_degg['dt'] = diff
    #print(df_degg.mfhTime.values)
    #print(diff)
    for it0, port in enumerate(df_degg.port.unique()):
        for it1, channel in enumerate([1]):
            _df = df_degg[(df_degg.port == port) & (df_degg.channel == channel)]
            port = int(port)
            channel = int(channel)
            print(port, channel)
            if spe == True:
                new_mask = make_laser_freq_mask_spe(_df.timestamp.values, rate)
            else:
                new_mask = make_laser_freq_mask_spe(_df.timestamp.values, rate)
                
            _df_dt_slice = _df.loc[new_mask]
            if it0 == 0 and it1 == 0:
                df_dt_slice = _df_dt_slice
            else:
                df_dt_slice = pd.concat([df_dt_slice, _df_dt_slice])
            print(f'Events remaining after deltaT cut: {(np.sum(new_mask)/len(new_mask))*100}%')

            
    #print(df_dt_slice.channel.unique())

    ##Matching is performed here!!!
    df_matched, ERROR_FLAG = findValidTriggers(df_dt_slice, df_ref)

    ##computation is expensive, so cache the df
    print("Created updated dataframe with matched trigger information")
    
#    mask = (diff > 0.0009e15) & (diff < 0.00111e15)
#    df_dt_slice = df_degg.loc[mask]
    
#    findValidTriggers(df_dt_slice, df_ref)
    ##computation is expensive, so cache the df
#    print("Created updated dataframe with matched trigger information")
    df_dt_slice.to_hdf('matched_triggers_test.hdf5', key='df', mode='w')

    return df_matched, ERROR_FLAG

def make_laser_freq_mask_spe(timestamps, rate=500, tolerance=5e-5,
                             smallOffset=5, pairRange=1000):
    laser_freq_in_hz = rate
    timestamps_per_second = 240e6
    dt_in_timestamps = timestamps_per_second / laser_freq_in_hz
    ##try to find laser triggers by checking delta-T
    starting_idx = -1
    num_pairs = 0
    starting_inds = []
    for i, t in enumerate(timestamps):
        for j in range(pairRange):
            if (i+j) >= len(timestamps):
                break
            delta = timestamps[i+j] - t
            if delta >= dt_in_timestamps - smallOffset and delta <= dt_in_timestamps + smallOffset:
                #print(i, j)
                #print(delta, delta/timestamps_per_second)
                num_pairs += 1
                starting_inds.append(i)
                if starting_idx == -1:
                    starting_idx = i
                    #break
        #if starting_idx != -1:
        #    break

    if starting_idx == -1:
        print('No valid index match for laser!')
        #raise ValueError('No starting index could be found!')
        return np.zeros_like(timestamps, dtype=bool)

    mask_list = []
    valid_range = 1000
    print(f'Number of starting pts: {len(starting_inds)}')
    for ind in tqdm(starting_inds):
        timestamps_shifted = timestamps - timestamps[ind]
        timestamps_in_dt = timestamps_shifted / dt_in_timestamps
        rounded_timestamps = np.round(timestamps_in_dt)
        mask_i = np.isclose(timestamps_in_dt, rounded_timestamps,
                       atol=tolerance, rtol=0)
        min_pt = ind-valid_range
        if min_pt < 0:
            min_pt = 0
        max_pt = ind+valid_range
        if max_pt > len(mask_i):
            max_pt = len(mask_i)-1
        mask_i[:min_pt] = 0
        mask_i[max_pt:] = 0
        mask_list.append(mask_i)

    master_mask = [False] * len(mask_list[0])
    for m in mask_list:
        master_mask = np.logical_or(master_mask, m)
    return master_mask

    ########
    new_mask = np.isclose(timestamps_in_dt, rounded_timestamps,
                       atol=tolerance, rtol=0)
    print(timestamps_in_dt)
    print(num_pairs, np.sum(new_mask))
    print(np.sum(new_mask)/len(timestamps))
    return new_mask

def findValidTriggers(df_degg, df_ref):
    ERROR_FLAG = ''

    t_max_degg = np.max(df_degg.mfhTime.values)
    t_min_degg = np.min(df_degg.mfhTime.values)
    t_ref = df_ref.mfhTime.values
    t_ind = df_ref.triggerNum.values
    drift_ref  = df_ref.clockDrift.values
    delay_ref = df_ref.cableDelay.values
    ref_delta = df_ref.delta.values[0]

    ##NOTE: for now ignoring the batching effect
    ##but anyway - batching just means some D-Egg triggers get thrown away, less than 1 in 200

    ##t_ref is already sorted
    ##sort dataframe based on mfhTime
    df_degg.sort_values(by='mfhTime', inplace=True)

    valid_row = [False] * len(df_degg.index)
    df_degg['t_match']          = np.zeros(len(df_degg.index))
    df_degg['matchInd']         = np.zeros(len(df_degg.index))
    df_degg['matchClockDrift']  = np.zeros(len(df_degg.index))
    df_degg['matchCableDelay1'] = np.zeros(len(df_degg.index))
    df_degg['matchCableDelay2'] = np.zeros(len(df_degg.index))
    df_degg['refDelta']         = np.zeros(len(df_degg.index))
    ##then do comparison
    ##looks like some of them need more than 100 ns -- due to the startup time?
    t_tolerance = 200e-9

    print('Matching Triggers')
    pmt_num = 0
    for port in df_degg.port.unique():
        for channel in [1]:
            _df_s = df_degg[(df_degg.port == port) & (df_degg.channel == channel)]
            if len(_df_s.index.values) == 0:
                print(f'Port {port} and channel {channel} are empty?')
                continue
            ##loop over each element in the dataframe
            i = 0
            matchList   = np.zeros(len(_df_s.delta.values))
            matchInd    = np.zeros(len(_df_s.delta.values))
            matchDrift  = np.zeros(len(_df_s.delta.values))
            matchDelay1 = np.zeros(len(_df_s.delta.values))
            matchDelay2 = np.zeros(len(_df_s.delta.values))
            validRow    = [False] * len(_df_s.delta.values)
            deltaList   = np.zeros(len(_df_s.delta.values))

            dummy_mask = [False] * len(_df_s.delta.values)

            ##only match within blocks
            for block in _df_s.blockNum.unique():
                _df_ref = df_ref[df_ref.blockNum == block]
                t_ref = _df_ref.mfhTime.values
                drift_ref  = _df_ref.clockDrift.values
                delay_ref = _df_ref.cableDelay.values
                ref_delta = _df_ref.delta.values[0]
                _df = _df_s[_df_s.blockNum == block]
                
                for num, (td, delta) in tqdm(enumerate(zip(_df.mfhTime.values, _df.delta.values))):
                    valid_row = False
                    ttList = (td - t_ref)/1e15 + (delta - df_ref.delta.values[0])
                    #print("ttList", type(ttList))
                    # print("(delta - df_ref.delta.values[0])", (delta - df_ref.delta.values[0])) 
                    # plt.figure()
                    # plt.scatter(range(len(ttList)), ttList)
                    # plt.xlabel('ttList', fontsize = 18)
                    # plt.show()
                    # plt.close()
                    if num == 0:
                        offset = np.max(ttList) 
                        print("offset", offset)
                        ttList = [i - offset for i in ttList]
#                         ax1.scatter(block, offset)
                    else:
                        ttList = [i - offset for i in ttList]
                   
                    ##this is now a mask over the reference df!
                    mask = np.abs(ttList) <= t_tolerance
                    itrue = np.argwhere(mask > 0)
                    #print(np.abs(ttList))
                    if np.sum(mask) == 1:
                        #print(td, t_ref, (td-t_ref), (td-t_ref)/1e15)
                        validRow[i]    = True
                        # matchList[i]   = t_ref[mask][0]
                        # matchInd[i]    = t_ind[mask][0]
                        # matchDrift[i]  = drift_ref[mask][0]
                        # matchDelay1[i] = delay_ref[mask][0][0]
                        # matchDelay2[i] = delay_ref[mask][0][1]
                        itrue = itrue[0][0]
                        matchList[i]   = float(t_ref[itrue]) + offset * 1e15
                        matchInd[i]    = t_ind[itrue]
                        matchDrift[i]  = drift_ref[itrue]
                        matchDelay1[i] = delay_ref[itrue][0]
                        matchDelay2[i] = delay_ref[itrue][1]
                        deltaList[i]   = ref_delta
                    elif np.sum(mask) == 0:
                        pass
                    else:
                        print(f'Sum: {np.sum(mask)} was greater than 1!')
                    i += 1

            if np.sum(validRow) == 0:
                warn_msg = f'No matches found in TTS analysis for {port}:{channel}! \n'
                warn_msg = warn_msg + 'Is the linearity data OK? If not, the fiber'
                warn_msg = warn_msg + ' transmission may be bad. New data may need to be collected.'
                warn_msg = warn_msg + ' Contact an expert immediately!'
                print(warn_msg)
                ERROR_FLAG = ERROR_FLAG + 'warn_msg'
                continue
                ##this error is preventing the analysis from finishing
                #raise ValueError(f'No matches at all!')

            _df_s['t_match'] = matchList
            _df_s['matchInd'] = matchInd
            _df_s['matchClockDrift'] = matchDrift
            _df_s['matchCableDelay1'] = matchDelay1
            _df_s['matchCableDelay2'] = matchDelay2
            _df_s['refDelta'] = deltaList
            _df_s['valid'] = validRow
            if pmt_num == 0:
                if len(_df_s.index.values) == 0:
                    raise ValueError(f'Size of _df_s is 0! ({_df_s.channel})')
                new_df = _df_s
            else:
                new_df = pd.concat([new_df, _df_s])
            pmt_num += 1

    return new_df, ERROR_FLAG

@click.command()
@click.argument('data_dir')

def main(data_dir):
	file_list = glob.glob(data_dir + "/total_00001_charge_stamp_*") 
	print("file length =", len(file_list))
	print(sorted(file_list))
	
	for file in file_list:
		r_point = file.split(".")[0].split("_")[-2]
		t_point = file.split(".")[0].split("_")[-1]
		print("r_point =", r_point, "t_point =", t_point)
		df = pd.read_hdf(file)
		df_=checkTabletop(df)
		try:
			test_df, errorflag = compareTimes(df,df_)
			df_degg =test_df[(test_df.type=="degg") & (test_df.channel == 1) & (test_df.valid==True)]
		except:
			test_df = df
			df_degg =test_df[(test_df.type=="degg") & (test_df.channel == 1)]
		#df_degg.charge.hist(bins=200)
		#plt.show()

		df_degg.to_hdf(data_dir + "/trigger/df_matched_trigger_" + r_point + "_" + t_point + ".hdf", key='df', mode='w')

if __name__ == "__main__":
	main()
