from GraphDecoder import GraphDecoder
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_and_plot_logical_error(d_list, p_list, T=1, num_trials=80):
    """
    Simulate logical error rates for different distances `d` and plot results.

    Args:
        d_list (list): List of code distances.
        p_list (list): List of physical error rates.
        T (int): Number of rounds.
        num_trials (int): Number of trials for each (d, p) pair.

    Returns:
        dict: Mapping from d to list of logical error rates.
    """

    results = {}

    total_steps = len(d_list) * len(p_list) * num_trials
    with tqdm(total=total_steps, desc="Simulating") as pbar:
        for d in d_list:
            logical_error_p = []
            logical_error_p_no_correction = []

            for p in p_list:

                logical_error = []
                for _ in range(num_trials):
                    decoder = GraphDecoder(d=int(d), T=T, p=p)
                    decoder.inject_errors()
                    list_x_error = decoder.check_data_qubit_errors_X()
                    # list_z_error = decoder.check_data_qubit_errors_Z()

                    error_graph_x, paths_x = decoder.make_error_graph(list_x_error.copy(), error_key="X")
                    matching_graph_x = decoder.matching_graph(error_graph_x, "X")
                    matches_x = decoder.matching(matching_graph_x, "X")
                    flips_x = decoder.calculate_qubit_flips(matches_x, paths_x, "X")
                    decoder.apply_corrections(flips_x)

                    logical_x = decoder.logical_operator_X()
                    # if logical_x == 1:
                    #     print()
                    logical_error.append(logical_x)

                    pbar.update(1)

                error_rate = sum(logical_error) / len(logical_error)
                logical_error_p.append(error_rate)


                logical_error = []
                for _ in range(num_trials):
                    decoder = GraphDecoder(d=int(d), T=T, p=p)
                    decoder.inject_errors()
                    decoder.check_data_qubit_errors_X()

                    logical_x = decoder.logical_operator_X()
                    logical_error.append(logical_x)

                    pbar.update(1)

                error_rate = sum(logical_error) / len(logical_error)
                logical_error_p_no_correction.append(error_rate)

            results[f"{d}"] = logical_error_p
            results[f"{d} - no correction"] = logical_error_p_no_correction

    # Plotting
    import pickle
    with open('results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    plt.figure(figsize=(10, 6))
    for d, logical_errors in results.items():
        plt.plot(p_list, logical_errors, label="d = " + d)

    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate vs Physical Error Rate for Different Distances d")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Benchmark.png")
    plt.show()

    return results

d_list = {5,7,9}
p_list = np.linspace(0.01, 1, 10)
simulate_and_plot_logical_error(d_list, p_list)
