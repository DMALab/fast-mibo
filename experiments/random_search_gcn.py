import argparse
import numpy as np

from fast_mibo.configs import generate_gcn_config
from fast_mibo.evaluations import evaluate_gcn
from time import process_time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--data_path", type=str, default="../data/cora")
parser.add_argument("--runcount", type=int, default=10)
parser.add_argument("--metric", type=str, default="accuracy")
args = parser.parse_args()

perfs = list()
times = list()
for run_id in range(args.runcount):
    t0 = process_time()
    gcn_config = generate_gcn_config()
    perf = evaluate_gcn(gcn_config, args)
    perfs.append(perf)
    times.append(process_time() - t0)
    print("run_id:", run_id, " perf: ", perf)

# np.save("random_search_gcn_perfs.npy", perfs)
# np.save("random_search_gcn_times.npy", times)
