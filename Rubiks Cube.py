import pycuber as pc
from ScrambleGenerator import FullScramble


# create a fully scrambled Cube
def fullyScrambledCube(myCube):

    # create a scramble
    my_scramble = pc.Formula(FullScramble())

    # scramble the cube
    myCube(my_scramble)
    return myCube


def main():
    # initialize the Cube
    mycube = pc.Cube()
    # print(repr(mycube))

    # scramble the Cube
    # mycube = fullyScrambledCube(mycube)

    # show the scrambled Cube
    print(mycube)
    cubelist = Listrepresentation(mycube)
    print(stringify(cubelist))
    print(pc.Cube(pc.array_to_cubies(stringify(cubelist))))


def stringify(intcube):
    """
    Converts a List of Ints to a String
    :param intcube: List of Integers
    :return: String of a Cube
    """
    strcube = ""
    for i in intcube:
        strcube = strcube + "% s" % i
    return strcube


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
            "D": ("DFL", "DF", "DFR", "DL", "D", "DR", "DBL", "DB", "DBR")  # "D" Face
        }.get(face)
        for location in cb:
            Cubelist.append(colorcode.get(str(cube[location][face])))
    return Cubelist


if __name__ == '__main__':
    main()
    # exit()
