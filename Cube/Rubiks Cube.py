import pycuber as pc
from ScrambleGenerator import Scramblegen
from NN import CreateDataset


# create a fully scrambled Cube
def fullyScrambledCube():
    scramble = Scramblegen()
    return scramble.FullScramble()


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


def main():
    # initialize the Cube
    mycube = pc.Cube()
    # print(repr(mycube))

    # mycube = fullyScrambledCube()

    # show the scrambled Cube
    print(mycube)
    cubelist = CreateDataset.Listrepresentation(mycube)
    print(f'cubelist: {cubelist}')
    # print(f'stringify: {stringify(cubelist)}')
    print(pc.Cube(pc.array_to_cubies(stringify(cubelist))))


if __name__ == '__main__':
    main()
