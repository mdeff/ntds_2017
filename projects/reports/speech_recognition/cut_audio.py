import numpy as np
import librosa
import librosa.display


def cut_signal( audio, Thresh = 0.1, mode = 'proxy',reach = 2000, number_lobes = 2):
    """Extracts relevant parts ofn audio signal.
       The Thresh i
nput value defines the sensitivity of the cut, its value has to be positive.
       Two modes can be chosen:
           - proxy(Default): Finds main energy lobe of the signal and also adds lobes that are within reach. 
                             The reach parameter can be adjusted adn has to be a positive value (default is 2000.)
           - num_lobes:      Finds the highest energy lobes of the signal. The parameter num_lobes (default value 2) 
                             defines how many of the largest lobes are being considered. """
    
    i_start, i_end, rmse_audio, shift = find_lobes(Thresh, audio)
    energy = np.array([])
    for i in range(len(i_start)):
        energy = np.append(energy,sum(rmse_audio[int(i_start[i]):int(i_end[i])]))
    
    if mode is 'num_lobes':
        lobes = np.argsort(energy)[-number_lobes:]
        start = np.min(i_start[lobes])
        end = np.max(i_end[lobes])
    elif mode is 'proxy':
        main_lobe = np.argsort(energy)[-1]
        start = i_start[main_lobe]
        end = i_end[main_lobe]
        
        for i in range(main_lobe):
            if (i_start[main_lobe]-i_end[i])<reach:
                start = np.min((i_start[i],start))
                
        for i in range(main_lobe,len(i_start)):   
            if (i_start[i]-i_end[main_lobe])<reach:
                end = i_end[i]
        
    else:
        print('ERROR: mode not implemented.')
    
        
    audio_cut = audio[int(np.max((0,int(start-shift-300)))):int(np.min((int(end)+300,len(audio))))]    
    return audio_cut

def find_lobes(Thresh, audio, shift = int(2048/16)):
    """Finds all energy lobes in an audio signal and returns their start and end indices. The parameter Thresh defines
       the sensitivity of the algforithm."""
    # Compute rmse
    audio = audio/np.max(audio)
    rmse_audio = librosa.feature.rmse(audio, hop_length = 1, frame_length=int(shift*2)).reshape(-1,)
    rmse_audio -= np.min(rmse_audio)
    i_start = np.array([])
    i_end = np.array([])
    for i in range(len(rmse_audio)-1):
        if (int(rmse_audio[i]>Thresh)-int(rmse_audio[i+1]>Thresh)) == -1:
            i_start = np.append(i_start,i)
        elif (int(rmse_audio[i]>Thresh)-int(rmse_audio[i+1]>Thresh)) == 1:    
            i_end = np.append(i_end,i)
    
    if len(i_start) == 0:
        i_start = np.append(i_start,0)
        
    if len(i_end) == 0:
        i_end = np.append(i_end,i)   
    
    if i_start[0]>i_end[0]:
        i_start = np.append(np.array(0), i_start)
        
    if i_start[-1]>i_end[-1]:
        i_end = np.append(i_end,i)
    
    return i_start, i_end, rmse_audio, shift