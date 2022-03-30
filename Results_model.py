import nengo
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.io import loadmat
import itertools
import scipy.io
from subset import subsets
from error import hearing_prediction, correct_prediction, detection_error_loss, distance_loss, sethares

#File Paths
full_path = "full_dataset_path.mat" #i.e., the full dataset
labels = "label_dataset_path.mat"

#Handling the data
matrix_data = loadmat(full_path)
label_data = loadmat(labels)
input_array = (np.log10(matrix_data["matrix"]))
input_array[input_array<0] = 0
output_array = (label_data["labels"])*3

#Global variables
nn= 30
initial = 0
final = initial+nn
D = nn
N = D*50

decoder_path = "file_from_trainable_model_path.mat"
decoder_data = loadmat(decoder_path)
decoders1 = decoder_data["decoders"]
decoders = decoders1[0,:,:]

sim_runtime = 5

new_indexes = subsets(output_array, nn)

with nengo.Network(seed=0) as net:

    # Desired output
    correct_stim = nengo.Node(nengo.processes.PresentInput(output_array[new_indexes, initial:final], presentation_time=0.15))

    # Hearing
    hearing_stim = nengo.Node(nengo.processes.PresentInput(input_array[new_indexes, initial:final], presentation_time=0.15))
    hearing_ensemble = nengo.Ensemble(N*4, D, radius=4, seed=0)
    decision_ensemble = nengo.Ensemble(N*5, D, radius=4, seed=0)

    #Connections
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
number_of_datapoints= round((sim_runtime/(0.15))-1)
hearing_sense = (hearing_prediction(sim.data[probe2]))[0:number_of_datapoints]
comparison = new_indexes[0:number_of_datapoints]
expectation = (correct_prediction(output_array[comparison]))


de_loss = (np.mean(detection_error_loss(expectation, hearing_sense))) #Detection-error loss
distance_loss = (distance_loss(expectation, hearing_sense)) #Loss based on the distance in semitones
consonance_loss = np.mean(sethares(expectation, hearing_sense)[0]) #Loss based on Plomp and Levelt's "Tonal Consonance"
consonance_loss1 = np.mean(sethares(expectation, hearing_sense)[1])

#Result Outputs

print(f"The accuracy under detection-error loss is {np.round((1-de_loss)*100, decimals= 1)}%")
print(f"The accuracy under distance loss is {np.round((1-(np.mean(distance_loss[0])))*100, decimals=1)}%")
print(f"The accuracy under linear distance loss is {np.round((1-(np.mean(distance_loss[1])))*100, decimals=1)}%")
print(f"The accuracy under consonance loss is {np.round((1-consonance_loss)*100, decimals=1)}%")
print(f"The accuracy under linear consonance loss is {np.round((1-consonance_loss1)*100, decimals=1)}%")



#Can plot the hearing signals vs Expected signals:

# plt.figure()
# plt.plot(sim.trange(), sim.data[probe4] , label="c")
# plt.plot(sim.trange(), sim.data[probe3] , label= "h")
# plt.xlabel("Time (s)")
# plt.ylabel("Magnitude")

# plt.show()
