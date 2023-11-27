# A main module to run various analysis with CFE model
# To implement sensitivity analysis with SALib. Currently this module supports Morris and Sobol analysis

# Import libraries
import multiprocessing as mp
from agent import Spotpy_Agent
import configparser
import time
from tqdm import tqdm


def map_with_progress(func, iterable, num_processes):
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(func, iterable), total=len(iterable)))
    pool.close()
    pool.join()
    return results


def main():
    start = time.perf_counter()

    config = configparser.ConfigParser()
    config.read("1_pre_calibration/config.ini")

    print(f"### Start {config['spotpy']['method']} calibration ###")
    spotpy_agent = Spotpy_Agent(config=config)

    # Implementation
    print("--- Evaluation started ---")
    spotpy_agent.run()
    spotpy_agent.finalize()

    # results = pool.map(salib_experiment.simulation, sampled_param_sets)
    print(f"--- Finished evaluation runs ---")

    end = time.perf_counter()

    print(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
