from Cube.ScrambleGenerator import Scramblegen
from Cube.CreateDataset import Listrepresentation
from NN.NNetwork import MyModel
import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
import sklearn.model_selection as skl
import pycuber as pc


# create a fully scrambled Cube
def get_solved_f2l():
    scramble = Scramblegen()
    return scramble.OLLScramble()


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


def getdata(loadup_test_data=False):
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
    oll_algos = pd.read_csv("../Cube/new_oll_algos.csv", dtype={'ID': int, 'Cube State': str,
                                                                'Formula': str, 'one-hot-vector': str})
    pll_algos = pd.read_csv("../Cube/new_pll_algos.csv", dtype={'ID': int, 'Cube State': str,
                                                                'Formula': str, 'one-hot-vector': str})

    # create a one-hot lookuptable for every possible case of oll and pll
    onehotollstr = oll_algos.iloc[:, 3].values
    onehotpllstr = pll_algos.iloc[:, 3].values
    onehotoll = stringtolistparser(onehotollstr, float)
    onehotpll = stringtolistparser(onehotpllstr, float)
    onehotolllookup = {oll_algos.values[i][1]: x for i, x in enumerate(onehotoll)}
    onehotplllookup = {pll_algos.values[i][1]: x for i, x in enumerate(onehotpll)}

    if loadup_test_data:
        # reads only loadup testdata
        loadup_testolls = pd.read_csv(path / "olls_1k.csv", dtype=typedict)
        loadup_testplls = pd.read_csv(path / "plls_1k.csv", dtype=typedict)

        # get loadup test data and Normalize
        loadup_testoll_matrix = loadup_testolls.iloc[:, :-1].values / 5
        loadup_testpll_matrix = loadup_testplls.iloc[:, :-1].values / 5
        loadup_testoll_vec = loadup_testolls.iloc[:, -1].values
        loadup_testpll_vec = loadup_testplls.iloc[:, -1].values

        # get the one-hot versions of the solution
        loadup_testoll_vector = [onehotolllookup[x] for x in loadup_testoll_vec]
        loadup_testpll_vector = [onehotplllookup[x] for x in loadup_testpll_vec]

        ollonehotlookup = {x: oll_algos.values[i][2] for i, x in enumerate(onehotollstr)}
        pllonehotlookup = {x: pll_algos.values[i][2] for i, x in enumerate(onehotpllstr)}

        return (
            np.array(loadup_testoll_matrix),
            np.array(loadup_testoll_vector),
            np.array(loadup_testpll_matrix),
            np.array(loadup_testpll_vector),
            ollonehotlookup,
            pllonehotlookup
        )

    else:
        # reads the csv with the according datatypes for training
        trainolls = pd.read_csv(path / "olls_10k.csv", dtype=typedict)
        trainplls = pd.read_csv(path / "plls_300k.csv", dtype=typedict)
        validationolls = pd.read_csv(path / "olls_1k.csv", dtype=typedict, )
        validationplls = pd.read_csv(path / "plls_10k.csv", dtype=typedict)

        # get training Data and Normalize
        trainingollmatrix = trainolls.iloc[:, :-1].values / 5
        trainingpllmatrix = trainplls.iloc[:, :-1].values / 5
        trainingollvec = trainolls.iloc[:, -1].values
        trainingpllvec = trainplls.iloc[:, -1].values

        # get validation data and Normalize
        validationollmatrix = validationolls.iloc[:, :-1].values / 5
        validationpllmatrix = validationplls.iloc[:, :-1].values / 5
        validationollvec = validationolls.iloc[:, -1].values
        validationpllvec = validationplls.iloc[:, -1].values

        # get the one-hot versions of the solution
        trainingollvector = [onehotolllookup[x] for x in trainingollvec]
        trainingpllvector = [onehotplllookup[x] for x in trainingpllvec]
        validationollvector = [onehotolllookup[x] for x in validationollvec]
        validationpllvector = [onehotplllookup[x] for x in validationpllvec]

        return (
            np.array(trainingollmatrix),
            np.array(trainingollvector),
            np.array(trainingpllmatrix),
            np.array(trainingpllvector),
            np.array(validationollmatrix),
            np.array(validationollvector),
            np.array(validationpllmatrix),
            np.array(validationpllvector),
        )


def accuracy(prediction, actual):
    predict = prediction.tolist()
    act = actual.tolist()
    matches = [i for i, j in zip(predict, act) if np.argmax(i) == np.argmax(j)]
    acc = len(matches) / len(act)
    return acc * 100


def one_hotify(predicted_y):
    assert len(predicted_y) == 1
    # print(predicted_y)
    index = np.argmax(predicted_y)
    # print(index)
    one_hot = np.zeros(len(predicted_y[0]))
    one_hot.itemset(index, 1.0)
    return one_hot


def fit_NN():
    (learnollmatrix, learnollvec, learnpllmatrix, learnpllvec,
     validationollmatrix, validationollvec, validationpllmatrix, validationpllvec
     ) = getdata(loadup_test_data=False)
    # print(f"learnollmatrix:\n{learnpllmatrix.shape}")
    # print(f"learnollvec:\n{learnpllvec.shape}")
    # print(f"validationollmatrix:\n{validationpllmatrix.shape}")
    # print(f"validationollvec:\n{validationpllvec.shape}")

    pll_savepath = Path("training_pll/pll.ckpt")
    oll_savepath = Path("training_oll/oll.ckpt")

    # k-fold cross validation
    kf = skl.KFold(n_splits=5, random_state=None, shuffle=False)
    ollmodel = MyModel()
    ollmodel.oll_compile()
    ollmodel()
    # training loop oll
    for train_index, test_index in kf.split(learnollmatrix):
        X_train, X_test = learnollmatrix[train_index], learnollmatrix[test_index]
        y_train, y_test = learnollvec[train_index], learnollvec[test_index]
        ollmodel.train(X_train, y_train, epochs=7)
        print()
        # predicted = ollmodel.predict(X_test)
        # acc = accuracy(predicted, y_test)
        # print(f"accuracy = {acc}%")

    pllmodel = MyModel()
    pllmodel.pll_compile()
    pllmodel()
    # training loop pll
    for train_index, test_index in kf.split(learnpllmatrix):
        X_train, X_test = learnpllmatrix[train_index], learnpllmatrix[test_index]
        y_train, y_test = learnpllvec[train_index], learnpllvec[test_index]
        pllmodel.train(X_train, y_train, epochs=5, verbose=1)
        print()
        # predicted = pllmodel.predict(X_test)
        # acc = accuracy(predicted, y_test)
        # print(f"accuracy = {acc}%")

    validation = ollmodel.predict(validationollmatrix)
    accurate = accuracy(validation, validationollvec)
    print(f"oll validation accuracy = {accurate}%")
    ollmodel.save(oll_savepath)

    validation = pllmodel.predict(validationpllmatrix)
    accurate = accuracy(validation, validationpllvec)
    print(f"pll validation accuracy = {accurate}%")
    pllmodel.save(pll_savepath)


def Use_NN():
    (testoll_X, testoll_y, testpll_X, testpll_y, oll_lookup, pll_lookup) = getdata(loadup_test_data=True)
    oll_savepath = Path("training_oll/oll.ckpt")
    pll_savepath = Path("training_pll/pll.ckpt")

    oll_nn = MyModel()
    oll_nn.oll_compile()
    # oll_nn()
    oll_nn.load(oll_savepath)
    oll_nn.eval(testoll_X, testoll_y)

    pll_nn = MyModel()
    pll_nn.pll_compile()
    # pll_nn()
    pll_nn.load(pll_savepath)
    pll_nn.eval(testpll_X, testpll_y)

    fullsolve = ""
    Cube = get_solved_f2l()
    print(f"scrambled Cube:\n{repr(Cube)}")
    cubelist = np.array([Listrepresentation(Cube)]) / 5
    oll_key = array2string(one_hotify(oll_nn.predict(cubelist)))
    oll_algo = oll_lookup[f"{oll_key}"]
    Cube(oll_algo)
    fullsolve += f"OLL: {oll_algo},  "
    print(repr(Cube))
    cubelist = np.array([Listrepresentation(Cube)]) / 5
    pll_key = array2string(one_hotify(pll_nn.predict(cubelist)))
    pll_algo = pll_lookup[f"{pll_key}"]
    Cube(pll_algo)
    cubelist = np.array([Listrepresentation(Cube)]) / 5
    pll_key = array2string(one_hotify(pll_nn.predict(cubelist)))
    last_step = pll_lookup[f"{pll_key}"]
    Cube(last_step)
    fullsolve += f"PLL: {pll_algo} {last_step}"
    solved_cube = pc.Cube()
    print(repr(solved_cube))
    if is_solved(Cube):
        print("We have successfully solved the Cube using the following sequence:")
        print(fullsolve)
        print(repr(Cube))
    else:
        print("The Cube has not been solved Correctly:")
        print(f"After applying the sequence {fullsolve}\nThe Cube looks like this:\n{repr(Cube)}")


def array2string(arr):
    out = "["
    for i in arr:
        if len(out) > 2:
            out += ", "
        out += f"{i}"
    out += "]"
    return out


def is_solved(Cube):
    for side in "LUFDRB":
        sample = Cube[side].facings[side]
        for square in sum(Cube.get_face(side), []):
            if square != sample:
                return False
    return True


def test():
    zero = np.zeros(2, dtype=float)
    normal = np.array([0.002, 1.001], dtype=float)
    print(repr(zero))
    print(repr(normal))


if __name__ == '__main__':
    # todo: create a nn for cross and f2l
    # test()
    Use_NN()
    # cubestring = stringify([1, 5, 1, 4, 0, 0, 3, 5, 3, 2, 5, 4, 3, 1, 3, 4, 4, 0, 2, 1, 3, 4, 2, 1, 4, 5, 1, 5, 0, 5, 2, 3, 2, 2, 2, 2, 0, 1, 0, 3, 4, 4, 4, 1, 3, 0, 0, 1, 2, 5, 0, 5, 3, 5])
    # cube = pc.Cube(pc.array_to_cubies(cubestring))
    # print(repr(cube))
    # print(repr(cube(pc.Formula("R F D2 R' U B' D2 R B D"))))
