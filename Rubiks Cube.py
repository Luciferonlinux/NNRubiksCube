import pycuber as pc
from ScrambleGenerator import FullScramble

# todo create dataset(s) for the ai to train on (0-20 moves)


# create a fully scrambled Cube
def fullyScrambledCube(myCube):

    # create a scramble
    my_scramble = pc.Formula(FullScramble())

    # scramble the cube
    myCube(my_scramble)

    return myCube


# represents the Cube as list datatype
def Listrepresentation(cube):

    colorcode = {
        "[r]": 1,
        "[g]": 2,
        "[o]": 3,
        "[b]": 4,
        "[y]": 5,
        "[w]": 6
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
        }.get(face)

        for location in cb:
            Cubelist.append(colorcode.get(str(cube[location][face])))

    return Cubelist


def main():
    # initialize the Cube
    mycube = pc.Cube()
    # print(repr(mycube))

    # scramble the Cube
    # mycube = fullyScrambledCube(mycube)

    # show the scrambled Cube
    print(mycube)
    print(Listrepresentation(mycube))


if __name__ == '__main__':
    main()
    exit()
