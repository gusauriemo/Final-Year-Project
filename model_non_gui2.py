import nengo
import nengo_loihi
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.io import loadmat
from nengo.dists import Choice
import itertools
from nengo.utils.ensemble import response_curves, tuning_curves
import scipy.io
from subset import subsets
from error import hearing_prediction, correct_prediction, detection_error_loss, distance_loss, sethares

#File Paths
full_path = "/Users/gustavo/Documents/Final Year Project/POLY3/full_poly3-A-Eb-C.mat" #i.e., the full dataset
labels = "/Users/gustavo/Documents/Final Year Project/POLY3/labels_poly3-A-Eb-C.mat"

#Handling the data
matrix_data = loadmat(full_path)
label_data = loadmat(labels)
input_array = (np.log10(matrix_data["matrix"]))
input_array[input_array<0] = 0
output_array = (label_data["labels"])*3


#print(output_array[(subsets(output_array, 30))]) # prints the new array with values up to index 30 (note 30)
#Global variables
nn= 30
initial = 0
final = initial+nn
D = nn
N = D*50

decoder_path = "/Users/gustavo/Documents/Final Year Project/codes/decoders_30_100s.mat"
decoder_data = loadmat(decoder_path)
decoders1 = decoder_data["decoders"]
decoders = decoders1[0,:,:]

sim_runtime = 5

new_indexes = subsets(output_array, nn)

#nengo_loihi.set_defaults() #https://www.nengo.ai/nengo-loihi/v1.1.0/overview.html


with nengo.Network(seed=0) as net:

    # Desired output
    correct_stim = nengo.Node(nengo.processes.PresentInput(output_array[new_indexes, initial:final], presentation_time=0.15))

    # Hearing
    hearing_stim = nengo.Node(nengo.processes.PresentInput(input_array[new_indexes, initial:final], presentation_time=0.15))
    hearing_ensemble = nengo.Ensemble(N*4, D, radius=4, seed=0)
    decision_ensemble = nengo.Ensemble(N*5, D, radius=4, seed=0)

    #Connections
    #nengo.Connection(hearing_ensemble[:D], hearing_ensemble[D:], synapse=0.2)
    nengo.Connection(hearing_stim, hearing_ensemble[:D])

    #Direct neurons to ensemble connection
    conn2 = nengo.Connection(hearing_ensemble.neurons, decision_ensemble, transform=decoders)


    #Probes
    probe1 = nengo.Probe(hearing_ensemble,  synapse=0.01)
    probe4 = nengo.Probe(decision_ensemble, synapse=0.01) 
    probe2 = nengo.Probe(decision_ensemble, synapse=0.01, sample_every=0.15) 
    """this probe gives a matrix of the output values every 0.15s. 
    Currently, since only 30 notes are being input, the number of columns is 30. 
    The number of rows will be determined by sim_runtime/0.15 (to the lower bound)"""
    probe3 = nengo.Probe(correct_stim, synapse=0.01)
    spike_probe = nengo.Probe(decision_ensemble.neurons)
    spike_probe2 = nengo.Probe(hearing_ensemble.neurons)


with nengo.Simulator(net) as sim:
    sim.run(sim_runtime)

    spike_count = np.sum(sim.data[spike_probe] > 0, axis=0) #total number of spikes in decision_ensemble during simulation
    spike_count2 = np.sum(sim.data[spike_probe] > 0, axis=0) #total number of spikes in hearing_ensemble

#Model Characteristics
number_spikes = np.sum(spike_count) + np.sum(spike_count2)
number_neurons = net.n_neurons
print(f"Total numbers of spikes: {number_spikes}")
print(f"Total number of neurons: {number_neurons}")
print(f"Average spikes per neuron per second: {np.round((number_spikes/number_neurons)/sim_runtime, decimals=2)}")


#Error Calculations
last= round((sim_runtime/(0.15))-1)
hearing_sense = (hearing_prediction(sim.data[probe2]))[0:last]
comparison = new_indexes[0:last]
expectation = (correct_prediction(output_array[comparison]))

#print(f"Output of hearing: {hearing_sense}") #Compare this output of indexes from hearing to the indexes from input sound

#print(f"Expected output: {expectation}")

de_loss = (np.mean(detection_error_loss(expectation, hearing_sense)))
distance_loss = (distance_loss(expectation, hearing_sense))
#distance_loss1 = np.mean(distance_loss(expectation, hearing_sense))
consonance_loss = np.mean(sethares(expectation, hearing_sense)[0])
consonance_loss1 = np.mean(sethares(expectation, hearing_sense)[1])

#Result Outputs

print(f"The accuracy under detection-error loss is {np.round((1-de_loss)*100, decimals= 1)}%")
print(f"The accuracy under distance loss is {np.round((1-(np.mean(distance_loss[0])))*100, decimals=1)}%")
print(f"The accuracy under linear distance loss is {np.round((1-(np.mean(distance_loss[1])))*100, decimals=1)}%")
print(f"The accuracy under consonance loss is {np.round((1-consonance_loss)*100, decimals=1)}%")
print(f"The accuracy under linear consonance loss is {np.round((1-consonance_loss1)*100, decimals=1)}%")





# plt.figure()
# plt.plot(sim.trange(), sim.data[probe4] , label="c")
# plt.plot(sim.trange(), sim.data[probe3] , label= "h")
# plt.xlabel("Time (s)")
# plt.ylabel("Magnitude")

# plt.show()