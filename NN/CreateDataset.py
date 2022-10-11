from Cube.ScrambleGenerator import Scramblegen
import pycuber.solver.cfop.pll as p
import pycuber.solver.cfop.oll as o
import pandas as pd
import time
from multiprocessing import Pool
from threading import Thread
from pathlib import Path
from sys import stdout


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


# noinspection PyUnusedLocal
def LL_pair(*args):
    scramble = Scramblegen()
    oll = scramble.OLLScramble()
    del scramble
    ocube = o.OLLSolver(oll)
    ollcase = ocube.recognise()
    # print(type(ollcase))
    # caselist = [x for x in ollcase]
    ocube.solve()
    pllcase = p.PLLSolver(ocube.cube).recognise()
    ollrep = Listrepresentation(oll)
    ollrep.append(ollcase)
    pllrep = Listrepresentation(ocube.cube)
    pllrep.append(pllcase)
    return ollrep, pllrep


def LL_scramble(count, path, write=True):
    """
    Make a .csv file containing scrambles, that can be solved using pll and what pll is used
    """
    done = False

    def pretty_loading_animation():
        start = time.perf_counter()
        animation = [
            " [===========               ]",
            " [ ===========              ]",
            " [  ===========             ]",
            " [   ===========            ]",
            " [    ===========           ]",
            " [     ===========          ]",
            " [      ===========         ]",
            " [       ===========        ]",
            " [        ===========       ]",
            " [         ===========      ]",
            " [          ===========     ]",
            " [           ===========    ]",
            " [            ===========   ]",
            " [             ===========  ]",
            " [              =========== ]",
            " [               ===========]",
            " [=               ==========]",
            " [==               =========]",
            " [===               ========]",
            " [====               =======]",
            " [=====               ======]",
            " [======               =====]",
            " [=======               ====]",
            " [========               ===]",
            " [=========               ==]",
            " [==========               =]",
        ]
        idx = 0
        while done is not True:
            now = time.perf_counter()
            stdout.write(f"\rloading {animation[idx % len(animation)]}   Time elapsed: {int(now - start)}s")
            idx += 1
            time.sleep(1)
            if idx == len(animation):
                idx = 0
        if done:
            stop = time.perf_counter()
            stdout.write(f"\rDone in {stop - start:.2f}\n")

    header = ["Square % s" % i if i < 54 else "Type" for i in range(55)]
    ollpath = path / f"olls_{int(count / 1000)}k.csv"
    pllpath = path / f"plls_{int(count / 1000)}k.csv"
    t = Thread(target=pretty_loading_animation)
    print(f"Generating {count} LL scrambles ...")
    t.start()
    with Pool() as pool:
        process = pool.imap_unordered(LL_pair, range(count))
        oll = []
        pll = []
        for x in process:
            i, j = x
            oll.append(i)
            pll.append(j)
        # odata = [i for i in oll]
        # pdata = [i for i in pll]
    olls = pd.DataFrame(oll, columns=header)
    plls = pd.DataFrame(pll, columns=header)
    # print(olls)
    # print("Writing to .csv ...")
    if write:
        olls.to_csv(
            header=True,
            path_or_buf=ollpath,
            sep=',',
            na_rep='n/a',
            index=False
        )
        plls.to_csv(
            header=True,
            path_or_buf=pllpath,
            sep=',',
            na_rep='n/a',
            index=False
        )
    done = True
    t.join()


if __name__ == '__main__':
    datasetpath = Path(__file__).parents[1] / "Datasets"
    start = time.perf_counter()
    LL_scramble(50000, datasetpath)
    end = time.perf_counter()
    print(f'Took {end - start:.2f} seconds')
