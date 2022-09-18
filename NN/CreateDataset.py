from Cube.ScrambleGenerator import Scramblegen
import pycuber.solver.cfop.pll as p
import pycuber.solver.cfop.oll as o
import pandas as pd
import time
from multiprocessing import Pool


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
    case = [p.PLLSolver(pll).recognise()]
    return Listrepresentation(pll), case


def OLL_pair(lol):
    scramble = Scramblegen()
    oll = scramble.OLLScramble()
    case = [o.OLLSolver(oll).recognise()]
    return Listrepresentation(oll), case


def pll_scramble(count, write=True):
    """
    Make a .csv file containing scrambles, that can be solved using pll and what pll is used
    """
    header = ['Cubelist', 'PLL Type']

    with Pool(8) as pool:
        process = pool.imap_unordered(PLL_pair, range(count))
        data = [i for i in process]
    plls = pd.DataFrame(data, columns=header)
    print(plls)

    if write:
        plls.to_csv(
            path_or_buf="Datasets/plls.csv",
            sep=',',
            na_rep='n/a',
            index=False
        )


def oll_scramble(count, write=True):
    """
    Make a .csv file containing scrambles, that can be solved using pll and what pll is used
    """
    header = ['Cubelist', 'OLL Type']

    with Pool(8) as pool:
        process = pool.imap_unordered(OLL_pair, range(count))
        data = [i for i in process]
    plls = pd.DataFrame(data, columns=header)
    print(plls)

    if write:
        plls.to_csv(
            path_or_buf="Datasets/olls.csv",
            sep=',',
            na_rep='n/a',
            index=False
        )


if __name__ == '__main__':
    start = time.perf_counter()
    oll_scramble(200)
    end = time.perf_counter()
    print(f'took {end - start} seconds')
