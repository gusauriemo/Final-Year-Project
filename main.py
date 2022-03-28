from note.noteSetter import file_names, shortFileNames, chord_nameToNote, whole_frequency_setter, chord_notes
from note.connections import third_layer_connections
from output_labels import labelling
import numpy as np
import time
import scipy.io

start = time.time()
print("hello")
folder = "/Users/gustavo/Documents/Final Year Project/UMA database/UMAPiano-DB-Poly-3/UMAPiano-DB-Poly-3-Eb"
size = (len(file_names(folder)))
location = file_names(folder)

I = np.zeros(size)
I2= np.zeros(88)
matrix = np.zeros((np.shape(I)[0], np.shape(I2)[0]))
names= []
indexedChordNotes = []

#print(chord_notes("/Users/gustavo/Documents/Final Year Project/UMA database/UMAPiano-DB-Poly-3/UMAPiano-DB-Poly-3-A/UMAPiano-DB-Eb1A1A2-PE-M.wav", 200))
print(size)
#the following function takes every file in the folder, analyses the audio and outputs an array for each containing the magnitude of each note within the audio file
for i in range(size):
    x = shortFileNames(folder)[i]
    x = x[12:len(x)-9] #this outputs the three notes in the file
    names.append(x) 
    #print(file_names(folder)[i])   
    #x_prime = (chord_nameToNote(x)) #this outputs the three frequencies in the file
    #matrix[i,:] = whole_frequency_setter(file_names(folder)[i])

#scipy.io.savemat('UMAPiano-DB-Poly-4-Eb_minus 9000.mat', mdict={'matrix': matrix, 'names':names})

end = time.time()
print(end - start)


#for i in (range(3)): #if we do (6), the next should be (6,n). If we do (n,m), the next one should be (m,p)
 #   num=500 #decision value for minimum magnitude for note consideration (very important!!)
  #  while len(chord_notes(location[i], num)) < 2:
   #     num=num-5
    #while len(chord_notes(location[i], num)) > 5:
     #   num=num+5
    #print(chord_notes(file_names(folder)[i], num))
    #indexedChordNotes.append((chord_notes(file_names(folder)[i], num)))
    #x = shortFileNames(folder)[i]
    #x = x[12:len(x)-9]
    #indexedChordNotes.append(x)

#print(indexedChordNotes)
#for n in range(0, 8):
 #   m=2*n+1
  #  print(chord_nameToNote(indexedChordNotes[m]))



third_layer_connections()


