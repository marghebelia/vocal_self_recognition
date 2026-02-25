import numpy as np

def Extract_ts_of_pitch_praat(Fname
                            , time_step = 0.001
                            , pitch_floor = 220
                            , pitch_ceiling = 1500
                            , max_nb_candidates = 15
                            , accuracy = 0
                            , silence_threshold = 0.03
                            , voicing_threshold = 0.45
                            , octave_cost = 0.01
                            , octave_jump_cost = 0.35
                            , voiced_unvoiced_cost = 0.14
                            #, harmonicity_threshold = None
                            ):
    """ 
        Extract pitch time series using praat for the file Fname

        Input: 
            Fname : input file name
            time_step : time step to use for the analysis in seconds
            pitch_floor : minimul pitch posible, in HZ
            pitch_ceiling : maximum pitch posible, in HZ
            accuracy : on or off

        See parameters here : https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__raw_ac____.html

        Also, use harmonicity threshold (from 0 to 1) to clean pitch estimation values.
        Return: times, f0

    """
    import os
    import parselmouth
    Fname = os.path.abspath(Fname)

    #execture script with parselmouth
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch_BB.praat")
    _, out = parselmouth.praat.run_file(script
                                    , Fname
                                    , str(time_step)
                                    , str(pitch_floor)
                                    , str(pitch_ceiling)
                                    , str(max_nb_candidates)
                                    , str(accuracy)
                                    , str(silence_threshold)
                                    , str(voicing_threshold)
                                    , str(octave_cost)
                                    , str(octave_jump_cost)
                                    , str(voiced_unvoiced_cost)
                                    #, str(harmonicity_threshold)
                                    , capture_output=True)                               

    # Analyser le script de sortie
    out = out.splitlines()
    times = []
    f0s = []

    for line in out:
        line = line.split()
        times.append(line[0])
        f0s.append(line[1])

    f0s = [np.nan if item == u'--undefined--' else float(item) for item in f0s]

    #If harmonicity threshold is defined, clean with harm thresh
#   if harmonicity_threshold:
#       times =  [float(x) for x in times]
#       f0s   = [float(x) for x in f0s]
#       
#       from bisect import bisect_left
#       harm_time, harm_vals = get_harmonicity_ts(Fname, time_step=time_step, normalise=True)
#
#       cleaned_vals  = []
#       cleaned_times = []
#       for index, time in enumerate(times):
#           idx_harm = bisect_left(harm_time, time)
#
#
#           if harmonicity_threshold < harm_vals[idx_harm-1]:
#               cleaned_vals.append(f0s[index])
#               cleaned_times.append(time)


#       times, f0s = cleaned_times, cleaned_vals

    return times, f0s

def get_HNR( signal, rate, time_step = 0, min_pitch = 140, 
             silence_threshold = .1, periods_per_window = 4.5 ):

    import peakutils as pu

    """
    Computes mean Harmonics-to-Noise ratio ( HNR ).
    The Harmonics-to-Noise ratio ( HNR ) is the ratio
    of the energy of a periodic signal, to the energy of the noise in the 
    signal, expressed in dB. This value is often used as a measure of 
    hoarseness in a person's voice. 
    A HNR of 0 dB means there is equal energy in harmonics and in noise. The 
    first step for HNR  determination of a signal, in the context of this 
    algorithm, is to set the maximum frequency allowable to the signal's 
    Nyquist  Frequency. Then the signal is segmented. Then for each frame, it
    calculates the normalized autocorrelation, or the 
    correlation of the signal  to a delayed copy of itself. The highest 
    peak is picked. If the height of this peak is larger than 
    the strength of the silent candidate, then the HNR for this frame is 
    calculated from that peak. The height of the peak corresponds to the energy
    of the periodic part of the signal. Once the HNR value has been calculated 
    for all voiced frames, the mean is taken from these values and returned.
    This algorithm is adapted from: 
    http://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf
    and from:
    https://github.com/praat/praat/blob/master/fon/Sound_to_Harmonicity.cpp
            
    Args:
        signal ( numpy.ndarray ): This is the signal the HNR will be calculated from.
        rate ( int ): This is the number of samples taken per second.
        time_step ( float ): ( optional, default value: 0.0 ) This is the measurement, in seconds, of time passing between each frame. The smaller the time_step, the more overlap that will occur. If 0 is supplied, the degree of oversampling will be equal to four.
        min_pitch ( float ): ( optional, default value: 75 ) This is the minimum value to be returned as pitch, which cannot be less than or equal to zero
        silence_threshold ( float ): ( optional, default value: 0.1 ) Frames that do not contain amplitudes above this threshold ( relative to the global maximum amplitude ), are considered silent.
        periods_per_window ( float ): ( optional, default value: 4.5 ) 4.5 is best for speech. The more periods contained per frame, the more the algorithm becomes sensitive to dynamic changes in the signal.
        
    Returns:
        float: The mean HNR of the signal expressed in dB.
        
    Raises:
        ValueError: min_pitch has to be greater than zero.
        ValueError: silence_threshold isn't in [ 0, 1 ].

    """
    #checking to make sure values are valid
    if min_pitch <= 0:
        raise ValueError( "min_pitch has to be greater than zero." )
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError( "silence_threshold isn't in [ 0, 1 ]." )
    #degree of overlap is four
    if time_step <= 0: time_step = ( periods_per_window / 4.0 ) / min_pitch 
                                   
    Nyquist_Frequency = rate / 2.0
    max_pitch = Nyquist_Frequency
    global_peak = np.max( abs( signal - signal.mean() ) ) 
    
    window_len = periods_per_window / float( min_pitch )
    
    #finding number of samples per frame and time_step
    frame_len = int( window_len * rate )
    t_len = int( time_step * rate )
    
    #segmenting signal, there has to be at least one frame
    num_frames = max( 1, int( len( signal ) / t_len + .5 ) ) 
    
    seg_signal = [ signal[ int( i * t_len ) : int( i  * t_len ) + frame_len ]  
                                           for i in range( num_frames + 1 ) ]

    #initializing list of candidates for HNR
    best_cands = []
    for index in range( len( seg_signal ) ):
        
        segment = seg_signal[ index ]
        #ignoring any potential empty segment
        if len( segment) > 0:
            window_len = len( segment ) / float( rate )
    
            #calculating autocorrelation, based off steps 3.2-3.10
            segment = segment - segment.mean()
            local_peak = np.max( abs( segment ) ) 
            if local_peak == 0 :
                best_cands.append( .5 )
            else:
                intensity = local_peak / global_peak 
                window = np.hanning( len( segment ) )
                segment *= window
               
                N = len( segment )
                nsampFFT = 2 ** int( np.log2( N ) + 1 )
                window  = np.hstack( (   window, np.zeros( nsampFFT - N ) ) ) 
                segment = np.hstack( (  segment, np.zeros( nsampFFT - N ) ) )
                x_fft = np.fft.fft( segment )
                r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
                r_a = r_a[ : N ]
                r_a = np.nan_to_num( r_a )
                
                x_fft = np.fft.fft( window )
                r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
                r_w = r_w[ : N ]
                r_w = np.nan_to_num( r_w )
                r_x = r_a / r_w
                
                r_x /= r_x[ 0 ]
                #creating an array of the points in time corresponding to the 
                #sampled autocorrelation of the signal ( r_x )
                time_array = np.linspace( 0, window_len, len( r_x ) )
                i = pu.indexes( r_x )
                max_values, max_places = r_x[ i ], time_array[ i ]
                max_place_poss = 1.0 / min_pitch
                min_place_poss = 1.0 / max_pitch
        
                max_values = max_values[ max_places >= min_place_poss ]
                max_places = max_places[ max_places >= min_place_poss ]
                
                max_values = max_values[ max_places <= max_place_poss ]
                max_places = max_places[ max_places <= max_place_poss ]
                
                for i in range( len( max_values ) ):
                    #reflecting values > 1 through 1.
                    if max_values[ i ] > 1.0 : 
                        max_values[ i ] = 1.0 / max_values[ i ]
                
                #eq. 23 and 24 with octave_cost, and voicing_threshold set to zero
                if len( max_values ) > 0:
                    strengths = [ max( max_values ), max( 0, 2 - ( intensity /
                                                            ( silence_threshold ) ) ) ]
                #if the maximum strength is the unvoiced candidate, then .5 
                #corresponds to HNR of 0
                    if np.argmax( strengths ):
                        best_cands.append( 0.5 )  
                    else:
                        best_cands.append( strengths[ 0 ] )
                else:
                    best_cands.append( 0.5 )
    
    best_cands = np.array( best_cands )
    best_cands = best_cands[ best_cands > 0.5 ]
    if len(best_cands) == 0:
        return 0
    #eq. 4
    best_cands = 10.0 * np.log10( best_cands / ( 1.0 - best_cands ) )
    best_candidate = np.mean( best_cands )
    return best_candidate

def get_RMS_over_time(audio_file, window_size = 512, in_db = True): # window_size = 1024
    """
    parameters:
        audio_file  : file to anlayse
        window_size : window size for FFT computing

        returns : time series with the RMS and the time
    
    warning : 
        this function only works for mono files
    """ 
    import glob
    from scikits.audiolab import wavread, aiffread
    from scipy import signal
    import numpy as np

    try:
        sound_in, fs, enc = aiffread(audio_file)
    except ValueError:
        sound_in, fs, enc = wavread(audio_file)

    begin = 0
    values = []
    time_tags = []
    while (begin + window_size) < len(sound_in):
        data = sound_in[begin : begin + window_size]
        time_tag = (begin + (window_size / 2)) / np.float(fs)
        
        values.append(get_rms_from_data(data, in_db = in_db))
        time_tags.append(time_tag)
        begin = begin + window_size

    return time_tags , values

