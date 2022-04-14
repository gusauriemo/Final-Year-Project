from note.noteSetter import file_names, shortFileNames, chord_nameToNote, whole_frequency_setter
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib

folder = "folder with sounds" #not in the repo
size = (len(file_names(folder)))
location = file_names(folder)

I = np.zeros(size)
I2= np.zeros(88)
matrix = np.zeros((np.shape(I)[0], np.shape(I2)[0]))
names= []
indexedChordNotes = []
x_axis = range(1,89)



#the following function takes every file in the folder, analyses the audio and outputs an array for each containing the magnitude of each note within the audio file
for i in range(size):
    x = shortFileNames(folder)[i]
    x = x[12:len(x)-9] #this outputs the three note names in the file
    names.append(x) 
    x_prime = (chord_nameToNote(x)) #this outputs the three frequencies in the file
    matrix[i,:] = whole_frequency_setter(file_names(folder)[i])

scipy.io.savemat('matrix name', mdict={'matrix': matrix, 'names':names})