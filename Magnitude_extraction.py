from note.noteSetter import file_names, shortFileNames, chord_nameToNote, whole_frequency_setter, chord_notes
from note.connections import third_layer_connections
from output_labels import labelling
import numpy as np
import time
import scipy.io

folder = "folder_with_all_the_wav_files"
size = (len(file_names(folder)))
location = file_names(folder)

I = np.zeros(size)
I2= np.zeros(88)
matrix = np.zeros((np.shape(I)[0], np.shape(I2)[0]))
labels= []
indexedChordNotes = []

for i in range(size):
    """ This function takes every file in the folder, 
    analyses the audio and outputs an array for each wav file containing the magnitude of each one of the 88 notes."""
    x = shortFileNames(folder)[i] #a string with the file name
    x = x[12:len(x)-9] #Cuts the string such that only the notes are present. -->Works for the naming convention of the UMA database
    labels.append(x)
    #x_prime = (chord_nameToNote(x)) #optionally takes this label x and outputs the frequency of the notes. E.g, [A4] would yield [440]
    matrix[i,:] = whole_frequency_setter(file_names(folder)[i])

scipy.io.savemat('your_matrix.mat', mdict={'matrix': matrix, 'labels':labels})
