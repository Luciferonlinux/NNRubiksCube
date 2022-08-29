from random import randint


def FullScramble():

    faceOptions = (0, 1, 2, 3, 4, 5)
    bad = True
    scramble = ""
    scrambleFaces = []

    # Generate moves until a 20 long sequence has no repetition
    while bad is True:
        scrambleFaces = []
        for i in range(20):
            face = randint(0, 5)
            scrambleFaces.append(faceOptions[face])

        for i in range(0, len(scrambleFaces) - 1):
            if scrambleFaces[i] == scrambleFaces[i+1]:
                bad = True
                break
            else:
                bad = False

    # Generate a String based on the above generated sequence
    for i in range(20):
        scramble = moveswitcher(scrambleFaces[i]) + " " + scramble
    return scramble


# Same as above, just with a custom length
def customScramble(length):

    faceOptions = (0, 1, 2, 3, 4, 5)
    bad = True
    scramble = ""
    scrambleFaces = []

    while bad:
        scrambleFaces = []
        for i in range(length):
            face = randint(0, 5)
            scrambleFaces.append(faceOptions[face])

        for i in range(0, len(scrambleFaces) - 1):
            if scrambleFaces[i] == scrambleFaces[i+1]:
                bad = True
                break
            else:
                bad = False

    for i in range(length):
        scramble = moveswitcher(scrambleFaces[i]) + " " + scramble
    return scramble


# replaces match-case from python 3.10
def moveswitcher(i):
    moveOptions = {
        0: "F",
        1: "F'",
        2: "F2",
        3: "R",
        4: "R'",
        5: "R2",
        6: "U",
        7: "U'",
        8: "U2",
        9: "B",
        10: "B'",
        11: "B2",
        12: "L",
        13: "L'",
        14: "L2",
        15: "D",
        16: "D'",
        17: "D2"
        }

    move = {
        0: moveOptions[randint(0, 2)],
        1: moveOptions[randint(3, 5)],
        2: moveOptions[randint(6, 8)],
        3: moveOptions[randint(9, 11)],
        4: moveOptions[randint(12, 14)],
        5: moveOptions[randint(15, 17)]
    }
    return move[i]
