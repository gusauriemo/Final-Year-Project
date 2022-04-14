def decoder_fun(sim_runtime, LR, epoch):
    """This model trains upon decoders"""
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

    full_path = "matrix path" #data matrix
    labels = "matrix path" #label matrix
    matrix_data = loadmat(full_path)
    label_data = loadmat(labels)

    decoder_path = "decoder path"+ str(sim_runtime)+"_"+str(epoch-1)+".mat"
    decoder_data = loadmat(decoder_path)
    decoders1 = decoder_data["decoders"]
    decoders = decoders1[0,:,:]

    nn= 30 #number of notes used
    initial = 0
    final = initial+nn

    input_array = (np.log10(matrix_data["matrix"]))
    input_array[input_array<0] = 0

    output_array = (label_data["labels"])*3

    D = nn #number of dimensions
    N = D*50 #smallest number of neurons per ensemble

    new_indexes = subsets(output_array, nn)

    simulation_runtime = sim_runtime


    with nengo.Network(seed=0) as net:
        #Correct signal - i.e., desired output
        correct_stim = nengo.Node(nengo.processes.PresentInput(output_array[new_indexes, initial:final], presentation_time=0.15))
        correct_ensemble = nengo.Ensemble(N*4, D, radius=4)

        #Input signal — hearing
        hearing_stim = nengo.Node(nengo.processes.PresentInput(input_array[new_indexes, initial:final], presentation_time=0.15))
        hearing_ensemble = nengo.Ensemble(N*4, D, radius=4, seed=0)
        decision_ensemble = nengo.Ensemble(N*5, D, radius=4, seed=0)


        # Connections
        nengo.Connection(correct_stim, correct_ensemble[:D])
        nengo.Connection(hearing_stim, hearing_ensemble[:D])
        
        learn_conn = nengo.Connection(hearing_ensemble.neurons, decision_ensemble, transform= decoders,  #can also use zero in the array instead of np.random.random(1)
            learning_rule_type = nengo.PES(learning_rate=LR)
            )

        error = nengo.Ensemble(N*5, D, radius = 4)

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

    scipy.io.savemat('decoders_30_' + str(simulation_runtime)+"_" +str(epoch) +'.mat', mdict={'decoders': decoders})

