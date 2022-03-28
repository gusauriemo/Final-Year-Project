import numpy as np
import math
import matplotlib.pyplot as plt

def alt_sigmoid(x):
    transformed = []
    for i in x:
        transformed.append(1/(1+math.exp(4.5-(0.9*i))))
    return transformed

def linear_error(x):
    transformed = []
    for i in x:
        transformed.append(i* (1/12))
    return transformed

def hearing_prediction(sim_data):
    """returns the indexes of the three highest-valued notes in the decision ensemble for simulated inputs.
    Takes as input the probe data from decision ensemble at 0.15s intervals
    E.g., hearing_prediction(sim.data[probe2]) """
    output_indexes = []
    for i in range(len(sim_data)):
        temp = sim_data[i,:]
        output_index = np.where((temp) == (max(temp)))[0]
        temp = np.delete((temp),(output_index))
        output_index2 = np.where((temp) == (max(temp)))[0]
        temp = np.delete((temp),(output_index2))
        output_index3 = np.where((temp) == (max(temp)))[0]
        np.delete((temp),(output_index3))
        all_indexes = [output_index[0], output_index2[0], output_index3[0]]
        output_indexes.append(all_indexes)
    return(output_indexes)


#print(hearing_prediction(sim.data[probe2])) #Compare this output of indexes from hearing to the indexes from input sound

def correct_prediction(array):
    """This function outputs the indexes of all correct notes for all notes in the labels matrix.
    Used for evaluating accuracy when compared against hearing_prediction function.
    Input is the output_array"""
    indexes = []
    for i in range(len(array)):
        index = np.where((array[i,:])>2)[0]
        indexes.append((index.tolist())) #this array (indexes) contains the index of every note in the chord 
    return indexes



def detection_error_loss(prediction, target):
    """ Both predition and ytarget are assumed to be matrices where each row has length 3"""
    bools= []
    for array in range(len(prediction)):
        for index in range(3):
            if prediction[array][index] == target[array][0]:
                bools.append(0)
            elif prediction[array][index] == target[array][1]:
                bools.append(0)
            elif prediction[array][index] == target[array][2]:
                bools.append(0)
            else:
                    bools.append(1)

    return bools


def distance_loss(prediction, target):
    differences = []
    for array in range(len(prediction)):
        for index in range(3):
            diff = np.abs(prediction[array][index] - target[array][0])
            diff= np.minimum(np.abs(prediction[array][index] - target[array][1]), diff)
            diff = np.minimum(np.abs(prediction[array][index] - target[array][2]), diff)
            differences.append(np.mod(diff,12))
    output = alt_sigmoid(differences)
    output2 = linear_error(differences)
    return output,output2
    #find the closest value in prediction to any of the targets and calculate the difference
    # define some sort of progressive error such that bigger distances have higher error



def sethares(prediction, target):
    differences = []
    consonance_rank = [(1,2), (2,3), (3,5), (3,4), (4,5), (5,8), (5,6), (5,7), (5,9), (8,9), (8,15), (15,16)]
    ranking = {0:0, 1:12, 2:10, 3:7, 4:5, 5:4, 6:8, 7:2, 8:6, 9:3, 10:11 ,11:9, 12:1}
    for array in range(len(prediction)):
        for index in range(3):
            diff = np.abs(prediction[array][index] - target[array][0])
            diff = ranking[np.mod(diff, 12)]
            diff2 = np.abs(prediction[array][index] - target[array][1])
            diff2= ranking[np.mod(diff2, 12)]
            diff3 = np.abs(prediction[array][index] - target[array][2])
            diff3 = ranking[np.mod(diff3, 12)]
            smallest = min(min(diff, diff2),diff3)
            differences.append(smallest)
    output = alt_sigmoid(differences)
    output2 = linear_error(differences)
    return output, output2


    #based on calculating the difference (as in the above function),
    # use the distances to set the error with the sethares method.

# axis= [x / 10.0 for x in range(0, 120)]
# plt.rcParams.update({'font.size': 15})
# plt.figure()
# plt.plot(axis, alt_sigmoid(axis) , label="c")
# plt.plot(axis, linear_error(axis) , label="c")
# plt.xlabel("Difference (in semitones)")
# plt.ylabel("Error")
# plt.show()
