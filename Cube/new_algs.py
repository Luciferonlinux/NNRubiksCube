import pandas as pd
import pycuber as pc
import tensorflow as tf


def main():
    oll_algos = pd.read_csv("oll_algos.csv", header=None, dtype=str).values
    pll_algos = pd.read_csv("pll_algos.csv", header=None, dtype=str).values

    def y_dict(sequence):
        dic = {"L": "B", "F": "L", "R": "F", "B": "R"}
        res = ""
        for char in sequence:
            res += dic[char]
        return res

    ids = 0
    algo_dict = []
    for name, rec_id, algo in pll_algos:
        for i in range(4):
            smth = rec_id[-3*i:] + rec_id[:-3*i]
            for _ in range(4):
                algo_dict.append([ids, smth, pc.Formula(algo).insert(0, pc.Step("U") * i)])
                smth = y_dict(smth)
                ids += 1
    rec_id = "LLLFFFRRRBBB"
    algo = []
    for i in range(4):
        algo_dict.append([ids, rec_id[-3 * i:] + rec_id[:-3 * i], pc.Formula(algo).insert(0, pc.Step("U") * i)])
        ids += 1

    indices = [i for i, _ in enumerate(algo_dict)]
    onehot = tf.one_hot(indices, len(algo_dict)).numpy().tolist()

    for i, _ in enumerate(algo_dict):
        algo_dict[i].append(onehot[i])

    new_pll_algos = pd.DataFrame(algo_dict, columns=('ID', 'Cube State', 'Formula', 'one-hot-vector'))
    new_pll_algos.to_csv(
        path_or_buf="new_pll_algos.csv",
        sep=",",
        na_rep="",
        columns=('ID', 'Cube State', 'Formula', 'one-hot-vector'),
        index=False
    )

    ids = 0
    algo_dict = []
    for line in oll_algos:
        for i in range(4):
            if len(algo_dict) < 2 or algo_dict[ids-1][1] != line[1][-3*i:] + line[1][:-3*i] != algo_dict[ids-2][1]:
                algo_dict.append([ids, line[1][-3*i:] + line[1][:-3*i], pc.Formula(line[2]).insert(0, pc.Step("U")*i)])
                ids += 1
    algo_dict.append([ids, "000000000000", pc.Formula()])

    indices = [i for i, _ in enumerate(algo_dict)]
    onehot = tf.one_hot(indices, len(algo_dict)).numpy().tolist()

    for i, _ in enumerate(algo_dict):
        algo_dict[i].append(onehot[i])

    new_pll_algos = pd.DataFrame(algo_dict, columns=('ID', 'Cube State', 'Formula', 'one-hot-vector'))
    new_pll_algos.to_csv(
        path_or_buf="new_oll_algos.csv",
        sep=",",
        na_rep="",
        columns=('ID', 'Cube State', 'Formula', 'one-hot-vector'),
        index=False
    )


if __name__ == '__main__':
    main()
