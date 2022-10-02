from Cube.ScrambleGenerator import Scramblegen
import pycuber.solver.cfop.pll as p
import pycuber.solver.cfop.oll as o
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
from pathlib import Path


def Listrepresentation(cube):
    """
    Creates a List with integers Representing a Cubes sides
    :param cube: Pc.Cube
    :return: List of ints Representing a Cubes sides
    """
    colorcode = {
        "[r]": 0,
        "[g]": 1,
        "[o]": 2,
        "[b]": 3,
        "[y]": 4,
        "[w]": 5
    }
    Cubelist = []
    for face in ("L", "F", "R", "B", "U", "D"):
        cb = {
            "L": ("LUB", "LU", "LUF", "LB", "L", "LF", "LBD", "LD", "LDF"),  # "L" Face
            "F": ("FUL", "FU", "FUR", "FL", "F", "FR", "FDL", "FD", "FDR"),  # "F" Face
            "R": ("RUF", "RU", "RUB", "RF", "R", "RB", "RFD", "RD", "RDB"),  # "R" Face
            "B": ("BUR", "BU", "BUL", "BR", "B", "BL", "BDR", "BD", "BDL"),  # "B" Face
            "U": ("UBL", "UB", "UBR", "UL", "U", "UR", "UFL", "UF", "UFR"),  # "U" Face
            "D": ("DFL", "DF", "DFR", "DL", "D", "DR", "DBL", "DB", "DBR")   # "D" Face
        }[face]
        for location in cb:
            Cubelist.append(colorcode[str(cube[location][face])])
    return Cubelist


def PLL_pair(lol):
    scramble = Scramblegen()
    pll = scramble.PLLScramble()
    del scramble
    case = p.PLLSolver(pll).recognise()
    listrep = Listrepresentation(pll)
    listrep.append(case)
    return np.array(listrep)


def OLL_pair(lol):
    scramble = Scramblegen()
    oll = scramble.OLLScramble()
    del scramble
    case = o.OLLSolver(oll).recognise()
    listrep = Listrepresentation(oll)
    listrep.append(case)
    return np.array(listrep)


def pll_scramble(count, path, write=True):
    """
    Make a .csv file containing scrambles, that can be solved using pll and what pll is used
    """
    header = ["Square % s" % i if i < 54 else "PLL Type" for i in range(55)]
    csvpath = path / "plls.csv"

    print(f"Generating {count} pll scrambles ...")
    with Pool() as pool:
        process = pool.imap_unordered(PLL_pair, range(count))
        data = [i for i in process]
    plls = pd.DataFrame(data, columns=header)
    # print(plls)
    print("Writing to .csv ...")
    if write:
        plls.to_csv(
            header=True,
            path_or_buf=csvpath,
            sep=',',
            na_rep='n/a',
            index=False
        )


def oll_scramble(count, path, write=True):
    """
    Make a .csv file containing scrambles, that can be solved using pll and what pll is used
    """
    header = ["Square % s" % i if i < 54 else "PLL Type" for i in range(55)]
    csvpath = path / "olls.csv"

    print(f"Generating {count} oll scrambles ...")
    with Pool() as pool:
        process = pool.imap_unordered(OLL_pair, range(count))
        data = [i for i in process]
    olls = pd.DataFrame(data, columns=header)
    # print(olls)
    print("Writing to .csv ...")
    if write:
        olls.to_csv(
            header=True,
            path_or_buf=csvpath,
            sep=',',
            na_rep='n/a',
            index=False
        )


if __name__ == '__main__':
    datasetpath = Path(__file__).parents[1] / "Datasets"
    start = time.perf_counter()
    pll_scramble(10, datasetpath)
    oll_scramble(10, datasetpath)
    end = time.perf_counter()
    print(f'took {end - start} seconds')
