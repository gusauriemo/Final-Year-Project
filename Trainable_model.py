import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.io import loadmat
from nengo.dists import Choice
import itertools
from nengo.utils.ensemble import response_curves, tuning_curves
import scipy.io
from subset import subsets


full_path = "dataset_input_directory"
labels = "label_dataset_directory"
matrix_data = loadmat(full_path)
label_data = loadmat(labels)

nn= 30 #number of notes used
initial = 0
final = initial+nn

input_array = (np.log10(matrix_data["matrix"]))
input_array[input_array<0] = 0

output_array = (label_data["labels"])*3 #increase magnitude of labels

D = nn #number of dimensions
N = D*50 #smallest number of neurons per ensemble

new_indexes = subsets(output_array, nn) #from subset file

simulation_runtime = 100 

#Important not to change seeds as they match the other model's
#Model:
with nengo.Network(seed=0) as net:
    #Correct signal - i.e., desired output
    correct_stim = nengo.Node(nengo.processes.PresentInput(output_array[new_indexes, initial:final], presentation_time=0.15))
    correct_ensemble = nengo.Ensemble(N*4, D, radius=4)

    #Input signal — hearing
    hearing_stim = nengo.Node(nengo.processes.PresentInput(input_array[new_indexes, initial:final], presentation_time=0.15))
    hearing_ensemble = nengo.Ensemble(N*4, D, radius=4, seed=0)
    decision_ensemble = nengo.Ensemble(N*5, D, radius=4, seed=0)
    error = nengo.Ensemble(N*5, D, radius = 4)

    # Connections
    nengo.Connection(correct_stim, correct_ensemble[:D])
    nengo.Connection(hearing_stim, hearing_ensemble[:D])
    learn_conn = nengo.Connection(hearing_ensemble, decision_ensemble,  #can also use zero in the array instead of np.random.random(1)
        learning_rule_type = nengo.PES(learning_rate=8e-5)
        )

    #Learning connections
    nengo.Connection(correct_ensemble, error, transform=-1)
    nengo.Connection(decision_ensemble, error, transform=1)
    nengo.Connection(error, learn_conn.learning_rule)

    #Probes

    probe1 = nengo.Probe(correct_ensemble, "decoded_output", synapse=0.01)
    probe2 = nengo.Probe(hearing_ensemble, "decoded_output", synapse=0.01)
    probe3 = nengo.Probe(error, synapse=0.01)
    probe4 = nengo.Probe(decision_ensemble, "decoded_output", synapse=0.01)
    probe_weights = nengo.Probe(learn_conn, "weights", synapse=0.01, sample_every= simulation_runtime) #obtains weights at the end of the simulation

with nengo.Simulator(net) as sim:
    sim.run(simulation_runtime)

decoders = (sim.data[probe_weights])

scipy.io.savemat('filename.mat', mdict={'decoders': decoders}) #Saving the decoder weigths

#once the simulation is over, we can use these decoders to simply run the simlation directly (no learning) and evaluate the loss for both training and testing


# Plot the input signals and decoded ensemble values

plt.figure()
plt.plot(sim.trange(), sim.data[probe1], label="c")
plt.plot(sim.trange(), sim.data[probe4],label="b")
plt.figure()
plt.plot(sim.trange(), sim.data[probe3],label="b")

plt.show()
