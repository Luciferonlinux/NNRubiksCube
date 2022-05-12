import pycuber as pc
from ScrambleGenerator import FullScramble

# Todo: Make Pattern Recognition for backpropagation (white cross)


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
    mycube = fullyScrambledCube(mycube)

    # show the scrambled Cube
    print(repr(mycube))
    return 1


if __name__ == '__main__':
    main()