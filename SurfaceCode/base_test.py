from GraphDecoder import GraphDecoder
import networkx as nx
import matplotlib.pyplot as plt
#TESTING
# Initialize decoder
logical_x = 0
use_decoder = True

d = 5
T = 1
p = 0.2
decoder = GraphDecoder(d=d, T=T, p = p)


'''
# 🧩 Visualize initial syndrome graphs
print("📌 Initial Syndrome Graph - X:")
node_index_x = decoder.graph_2D(decoder.S["X"], edge_label="distance")

print("📌 Initial Syndrome Graph - Z:")
node_index_z =  decoder.graph_2D(decoder.S["Z"], edge_label="distance")

print("📌 Initial Data Qubit Graph:")
decoder.draw_nodes_by_type(decoder.D)

decoder.D.nodes[(-2,0.0,0.0)]["bit_flip"] = True
decoder.D.nodes[(-2,3.0,3.0)]["bit_flip"] = True
'''


decoder.inject_errors() # Inject errors into the data qubit graph (only bit flips)
list_x_error = decoder.check_data_qubit_errors_X()
# list_z_error = decoder.check_data_qubit_errors_Z()

# Build the error graph for the selected syndrome locations
if use_decoder:
    error_graph_x, paths_x = decoder.make_error_graph(list_x_error.copy(), error_key="X", err_prob=p)
    nx.draw(error_graph_x)
    plt.draw()
    plt.savefig("error_graph_x.png")

    '''
    # Draw the weighted graph like in the image
    print("🎯 Error Graph from Syndromes (X errors):")
    decoder.graph_2D(error_graph_x, edge_label="weight")
    '''

    # Create the matching graph (adds virtuals if needed)
    matching_graph_x = decoder.matching_graph(error_graph_x, "X")
    # nx.draw(matching_graph_x)
    # plt.draw()
    # plt.savefig("matching_graph_x.png")

    '''
    # Draw matching graph
    print("🔗 Matching Graph (for MWPM):")
    decoder.graph_2D(matching_graph_x, edge_label="weight")
    '''


    # Perform matching
    matches_x = decoder.matching(matching_graph_x, "X")

    '''
    print("🎯 Matches from MWPM:")
    print(matches_x)
    '''
    # nx.draw(decoder.QX)
    # plt.draw()
    # plt.savefig("QX1.png")

    # Get shortest paths (decoded corrections)

    flips_x = decoder.calculate_qubit_flips(matches_x, paths_x, "X")

    # nx.draw(decoder.QX)
    # plt.draw()
    # plt.savefig("QX2.png")

    '''
    print("🎯 Physical qubit flips from MWPM:")
    print(flips_x)
    '''
    
    decoder.apply_corrections(flips_x)

    logical_x = decoder.logical_operator_X()
    print("🔑 Logical X operator:")
    print(logical_x)
    decoder.draw_nodes_by_type(decoder.D)

