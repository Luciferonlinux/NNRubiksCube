from Cube.ScrambleGenerator import Scramblegen
import pycuber.solver.cfop.pll as p
import pycuber.solver.cfop.oll as o
import pycuber.solver.cfop.cross as c
import pycuber.solver.cfop.f2l as f
from pycuber.formula import Formula
import pandas as pd
import time
from multiprocessing import Pool
from threading import Thread
from pathlib import Path
from sys import stdout


def Listrepresentation(cube):
    """
    Creates a List with integers Representing a Cubes sides
    args:
        cube: Pc.Cube

        returns: List of ints Representing a Cubes sides
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
    ollrep = Listrepresentation(oll)
    ocube = o.OLLSolver(oll)
    ollcase = ocube.recognise()
    ocube.solve()
    pllcase = p.PLLSolver(ocube.cube).recognise()
    ollrep.append(ollcase)
    pllrep = Listrepresentation(ocube.cube)
    pllrep.append(pllcase)
    return ollrep, pllrep

#
# def F2_pair(*args):
#     scramble = Scramblegen()
#     cross = scramble.FullScramble()
#     del scramble
#     crossrep = Listrepresentation(cross)
#     crosscube = c.CrossSolver(cross)
#     crosssolve = crosscube.solve()
#     print(f"{crosssolve=}")
#     f2lrep = Listrepresentation(crosscube.cube)
#     f2lcube = f.F2LSolver(crosscube.cube)
#     f2lsolve = Formula()
#     for _, x in enumerate(f2lcube.solve()):
#         f2lsolve += x[1]
#     crossrep.append(crosssolve)
#     f2lrep.append(f2lsolve)
#     return crossrep, f2lrep
#
#
# def F2_scramble(count, path, write=True, processes=16):
#     """
#     Make a .csv file containing scrambles, that can be solved using pll and what pll is used
#     """
#     done = False
#
#     def pretty_loading_animation():
#         start = time.perf_counter()
#         animation = [
#             " [===========               ]",
#             " [ ===========              ]",
#             " [  ===========             ]",
#             " [   ===========            ]",
#             " [    ===========           ]",
#             " [     ===========          ]",
#             " [      ===========         ]",
#             " [       ===========        ]",
#             " [        ===========       ]",
#             " [         ===========      ]",
#             " [          ===========     ]",
#             " [           ===========    ]",
#             " [            ===========   ]",
#             " [             ===========  ]",
#             " [              =========== ]",
#             " [               ===========]",
#             " [=               ==========]",
#             " [==               =========]",
#             " [===               ========]",
#             " [====               =======]",
#             " [=====               ======]",
#             " [======               =====]",
#             " [=======               ====]",
#             " [========               ===]",
#             " [=========               ==]",
#             " [==========               =]",
#         ]
#         idx = 0
#         while done is not True:
#             now = time.perf_counter()
#             stdout.write(f"\rloading {animation[idx % len(animation)]}   Time elapsed: {int(now - start)}s")
#             idx += 1
#             time.sleep(1)
#             if idx == len(animation):
#                 idx = 0
#         if done:
#             stop = time.perf_counter()
#             stdout.write(f"\rDone in {stop - start:.2f}s\n")
#
#     header = ["Square % s" % i if i < 54 else "Type" for i in range(55)]
#     crosspath = path / f"cross_{int(count / 1000)}k.csv"
#     f2lpath = path / f"f2l_{int(count / 1000)}k.csv"
#     t = Thread(target=pretty_loading_animation)
#     print(f"Generating {count} LL scrambles ...")
#     t.start()
#     with Pool(processes) as pool:
#         process = pool.imap_unordered(F2_pair, range(count))
#         cross = []
#         f2l = []
#         for x in process:
#             i, j = x
#             cross.append(i)
#             f2l.append(j)
#         # odata = [i for i in oll]
#         # pdata = [i for i in pll]
#     crosss = pd.DataFrame(cross, columns=header)
#     f2ls = pd.DataFrame(f2l, columns=header)
#     # print(olls)
#     # print("Writing to .csv ...")
#     if write:
#         crosss.to_csv(
#             header=True,
#             path_or_buf=crosspath,
#             sep=',',
#             na_rep='n/a',
#             index=False
#         )
#         f2ls.to_csv(
#             header=True,
#             path_or_buf=f2lpath,
#             sep=',',
#             na_rep='n/a',
#             index=False
#         )
#     done = True
#     t.join()


def LL_scramble(count, path, write=True, processes=16):
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
            stdout.write(f"\rDone in {stop - start:.2f}s\n")

    header = ["Square % s" % i if i < 54 else "Type" for i in range(55)]
    ollpath = path / f"olls_{int(count / 1000)}k.csv"
    pllpath = path / f"plls_{int(count / 1000)}k.csv"
    t = Thread(target=pretty_loading_animation)
    print(f"Generating {count} LL scrambles ...")
    t.start()
    with Pool(processes) as pool:
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
    # F2_scramble(1000, datasetpath, processes=16)
    # print(repr(F2_pair()))
    end = time.perf_counter()
    # print(f'Took {end - start:.2f} seconds')
