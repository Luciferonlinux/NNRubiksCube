from ScrambleGenerator import Scramblegen
from NN.NNetwork import MyModel
import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf


# create a fully scrambled Cube
def fullyScrambledCube():
    scramble = Scramblegen()
    return scramble.FullScramble()


def stringify(intcube):
    """
    Converts a List of Ints to a String

    Args:
        intcube: List of Integers

    Returns:
        String of a Cube
    """
    strcube = ""
    for i in intcube:
        strcube = strcube + "% s" % i
    return strcube


def getdata():
    path = Path(__file__).parents[1] / "Datasets"
    typedict = {
        'Square 0': int, 'Square 1': int, 'Square 2': int, 'Square 3': int, 'Square 4': int,
        'Square 5': int, 'Square 6': int, 'Square 7': int, 'Square 8': int, 'Square 9': int,
        'Square 10': int, 'Square 11': int, 'Square 12': int, 'Square 13': int, 'Square 14': int,
        'Square 15': int, 'Square 16': int, 'Square 17': int, 'Square 18': int, 'Square 19': int,
        'Square 20': int, 'Square 21': int, 'Square 22': int, 'Square 23': int, 'Square 24': int,
        'Square 25': int, 'Square 26': int, 'Square 27': int, 'Square 28': int, 'Square 29': int,
        'Square 30': int, 'Square 31': int, 'Square 32': int, 'Square 33': int, 'Square 34': int,
        'Square 35': int, 'Square 36': int, 'Square 37': int, 'Square 38': int, 'Square 39': int,
        'Square 40': int, 'Square 41': int, 'Square 42': int, 'Square 43': int, 'Square 44': int,
        'Square 45': int, 'Square 46': int, 'Square 47': int, 'Square 48': int, 'Square 49': int,
        'Square 50': int, 'Square 51': int, 'Square 52': int, 'Square 53': int, 'Type': str
    }
    # reads the csv with the accoring datatypes
    trainolls = pd.read_csv(path / "olls_1k.csv", dtype=typedict)
    trainplls = pd.read_csv(path / "plls_1k.csv", dtype=typedict)
    validationolls = pd.read_csv(path / "olls_0k.csv", dtype=typedict)
    validationplls = pd.read_csv(path / "plls_0k.csv", dtype=typedict)
    oll_algos = pd.read_csv("new_oll_algos.csv", dtype={'ID': int, 'Cube State': str,
                                                        'Formula': str, 'one-hot-vector': str})
    pll_algos = pd.read_csv("new_pll_algos.csv", dtype={'ID': int, 'Cube State': str,
                                                        'Formula': str, 'one-hot-vector': str})

    trainingollmatrix = trainolls.iloc[:, :-1].values
    trainingpllmatrix = trainplls.iloc[:, :-1].values
    trainingollvec = trainolls.iloc[:, -1].values
    trainingpllvec = trainplls.iloc[:, -1].values

    validationollmatrix = validationolls.iloc[:, :-1].values
    validationpllmatrix = validationplls.iloc[:, :-1].values
    validationollvec = validationolls.iloc[:, -1].values
    validationpllvec = validationplls.iloc[:, -1].values

    def stringtolistparser(vector, dtype):
        assert dtype is int or float
        out = []
        for vec in vector:
            stripped = vec.strip('][').split(', ')
            result = []
            for x in stripped:
                result.append(dtype(x))
            out.append(result)
        return out

    onehotollstr = oll_algos.iloc[:, 3].values
    onehotpllstr = pll_algos.iloc[:, 3].values
    onehotoll = stringtolistparser(onehotollstr, float)
    onehotpll = stringtolistparser(onehotpllstr, float)
    onehotolllookup = {oll_algos.values[i][1]: x for i, x in enumerate(onehotoll)}
    onehotplllookup = {pll_algos.values[i][1]: x for i, x in enumerate(onehotpll)}

    trainingollvector = [onehotolllookup[x] for x in trainingollvec]
    trainingpllvector = [onehotplllookup[x] for x in trainingpllvec]
    validationollvector = [onehotolllookup[x] for x in validationollvec]
    validationpllvector = [onehotplllookup[x] for x in validationpllvec]

    # splits = np.array_split(trainingollmatrix, 2)
    # for x in splits:
    #     pass

    return (
        np.array(trainingollmatrix),
        np.array(trainingollvector),
        np.array(trainingpllmatrix),
        np.array(trainingpllvector),
        np.array(validationollmatrix),
        np.array(validationollvector),
        np.array(validationpllmatrix),
        np.array(validationpllvector)
    )


def accuracy(prediction, actual):
    predict = prediction.tolist()
    act = actual.tolist()

    matches = [i for i, j in zip(predict, act) if tf.argmax(i).numpy() == tf.argmax(j).numpy]
    acc = len(matches) / len(act)
    return acc * 100


def main():

    (
        learnollmatrix, learnollvec, learnpllmatrix, learnpllvec,
        testollmatrix, testollvec, testpllmatrix, testpllvec
     ) = getdata()

    print(f"learnollmatrix:\n{learnollmatrix.shape}")
    print(f"learnollvec:\n{learnollvec.shape}")
    print(f"testollmatrix:\n{testollmatrix.shape}")
    print(f"testollvec:\n{testollvec.shape}")
    model = MyModel()
    model.oll_compile()
    model.train(learnollmatrix, learnollvec)
    predicted = model.predict(testollmatrix)
    acc = accuracy(predicted, testollvec)
    print(f"accuracy = {acc}%")
    # print(predicted, testollvec)


if __name__ == '__main__':
    # todo train nn
    # todo test nn
    main()
