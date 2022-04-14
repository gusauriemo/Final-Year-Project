import matplotlib.pyplot as plot
from pydub import AudioSegment
from scipy.io import wavfile as wav
import numpy as np
from os import listdir
from os.path import isfile, join

notes=[] #array for wav file notes
arr=[] #piano notes array

for x in range(88):
    arr.append(27.5*(2**(x/12))) #piano notes by utilising the interval ratios
  
set_frequencies = np.round(arr, decimals=2) #all piano notes rounded to 2 decimal places


def file_names(directory):
    """retrieve the names of all files and appends the full directory for easier indexing"""
    full_files=[] #array for full file names (which will go into chord_notes function)
    files = [f for f in listdir(directory) if isfile(join(directory, f)) if not f.startswith('.')]
    for i in range(len(files)):
        full_files.append(directory + "/" + files[i])
    return full_files

def shortFileNames(directory1): 
    """function to call the file name - used to get note names in main-py"""
    files = [f for f in listdir(directory1) if isfile(join(directory1, f))]    
    return files

def splitter(string): 
    """Receives a string and splits it where there is a space. Each of the parts will become an element of a List."""
    split = list(string.split(" "))
    return split


noteDictionary= {'A0': '27.5', 'A1': '55.0', 'A2': '110.0', 'A3': '220.0', 'A4': '440.0', 'A5': '880.0', 'A6': '1760.0', 'A7': '3520.0',
'A#0': '29.14', 'A#1': '58.27', 'A#2': '116.54', 'A#3': '233.08', 'A#4': '466.16', 'A#5': '932.33', 'A#6': '1864.66', 'A#7': '3729.31',
'Bb0': '29.14', 'Bb1': '58.27', 'Bb2': '116.54', 'Bb3': '233.08', 'Bb4': '466.16', 'Bb5': '932.33', 'Bb6': '1864.66', 'Bb7': '3729.31',
'B0': '30.87', 'B1': '61.74', 'B2': '123.47', 'B3': '246.94', 'B4': '493.88', 'B5': '987.77', 'B6': '1975.53', 'B7': '3951.07',
'B#0': '32.7', 'B#1': '65.41', 'B#2': '130.81', 'B#3': '261.63', 'B#4': '523.25', 'B#5': '1046.5', 'B#6': '2093.0', 'B#7': '4186.01',
'Cb0': '30.87', 'Cb1': '61.74', 'Cb2': '123.47', 'Cb3': '246.94', 'Cb4': '493.88', 'Cb5': '987.77', 'Cb6': '1975.53', 'Cb7': '3951.07',
'C1': '32.7', 'C2': '65.41', 'C3': '130.81', 'C4': '261.63', 'C5': '523.25', 'C6': '1046.5', 'C7': '2093.0', 'C8': '4186.01',
'C#1': '34.65', 'C#2': '69.3', 'C#3': '138.59', 'C#4': '277.18', 'C#5': '554.37', 'C#6': '1108.73', 'C#7': '2217.46',
'Db1': '34.65', 'Db2': '69.3', 'Db3': '138.59', 'Db4': '277.18', 'Db5': '554.37', 'Db6': '1108.73', 'Db7': '2217.46',
'D1': '36.71', 'D2': '73.42', 'D3': '146.83', 'D4': '293.66', 'D5': '587.33', 'D6': '1174.66', 'D7': '2349.32',
'D#1': '38.89', 'D#2': '77.78', 'D#3': '155.56', 'D#4': '311.13', 'D#5': '622.25', 'D#6': '1244.51', 'D#7': '2489.02',
'Eb1': '38.89', 'Eb2': '77.78', 'Eb3': '155.56', 'Eb4': '311.13', 'Eb5': '622.25', 'Eb6': '1244.51', 'Eb7': '2489.02',
'E1': '41.2', 'E2': '82.41', 'E3': '164.81', 'E4': '329.63', 'E5': '659.26', 'E6': '1318.51', 'E7': '2637.02',
'E#1': '43.65', 'E#2': '87.31', 'E#3': '174.61', 'E#4': '349.23', 'E#5': '698.46', 'E#6': '1396.91', 'E#7': '2793.83',
'Fb1': '41.2', 'Fb2': '82.41', 'Fb3': '164.81', 'Fb4': '329.63', 'Fb5': '659.26', 'Fb6': '1318.51', 'Fb7': '2637.02',
'F1': '43.65', 'F2': '87.31', 'F3': '174.61', 'F4': '349.23', 'F5': '698.46', 'F6': '1396.91', 'F7': '2793.83',
'F#1': '46.25', 'F#2': '92.5', 'F#3': '185.0', 'F#4': '369.99', 'F#5': '739.99', 'F#6': '1479.98', 'F#7': '2959.96',
'Gb1': '46.25', 'Gb2': '92.5', 'Gb3': '185.0', 'Gb4': '369.99', 'Gb5': '739.99', 'Gb6': '1479.98', 'Gb7': '2959.96',
'G1': '49.0', 'G2': '98.0', 'G3': '196.0', 'G4': '392.0', 'G5': '783.99', 'G6': '1567.98', 'G7': '3135.96',
'G#1': '51.91', 'G#2': '103.83', 'G#3': '207.65', 'G#4': '415.3', 'G#5': '830.61', 'G#6': '1661.22', 'G#7': '3322.44',
'Ab1': '51.91', 'Ab2': '103.83', 'Ab3': '207.65', 'Ab4': '415.3', 'Ab5': '830.61', 'Ab6': '1661.22', 'Ab7': '3322.44'}


def chord_nameToNote(chordName): 
    """For any given note name (or chord name in the array format of the database), outputs the frequency of those notes as an array. 
    Labelling of the data
    i.e., we could compare this output with the SNNs output"""
    for name, freq in noteDictionary.items():
        chordName = chordName.replace(name, freq + " ")
    outFrequencies = splitter(chordName)
    outFrequencies.remove('')
    outFrequencies = list(map(float, outFrequencies))
    return outFrequencies


#returns an array with the magnitude of each note (88 notes in order)
def whole_frequency_setter(fileName2):
    """Takes an audio file's directory as input and returns an array with the magnitude of each note.
    88 frequency bins."""
    samplingFrequency, signalData = wav.read(fileName2)

    graph = plot.magnitude_spectrum(signalData,Fs=samplingFrequency)
    y1values = graph[0]
    x1values = graph[1]
    limit = np.amax(np.where(x1values < 4200)[0]) #this simply limits the extent of the frequency spectrum search to the range we are actually interested in
    frequencies = x1values[:limit]
    magnitudes= y1values[:limit]
    plot.plot(x1values[:limit], y1values[:limit])
    magnitude_values = []
    plot.show()
    array = (np.where(((frequencies/set_frequencies[0])<((2**(0.5/12)))) & ((frequencies/set_frequencies[0])>=(2-(2**(0.5/12)))), set_frequencies[0], 0)) #first note
    for i in range(88) :
        output = np.where(((frequencies/set_frequencies[i])<((2**(0.5/12)))) & ((frequencies/set_frequencies[i])>(2-(2**(0.5/12)))), set_frequencies[i], 0) #half note tolerance for bin
        #here the np.where is considering all of the frequencies in the array from the x1values and comparing them against the notes, so that we "bin" them
        # We need now to attach a magnitude to these notes (this magnitude is store in y1values with the same indexes)
        indexes2 = np.where(output==set_frequencies[i])[0]
        #here we obtain the indexes for a given note "match" i.e., the indexes in output where the given set_frequencies[i] is found
        magnitude_values.append(sum(y1values[indexes2]))
        #summing all of the magnitudes of where there is a note-match i.e., the overall magnitude for that note-bin
    return magnitude_values #ordered by notes