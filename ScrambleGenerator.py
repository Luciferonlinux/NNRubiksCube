from random import randint


def FullScramble():

    faceOptions = (0, 1, 2, 3, 4, 5)
    moveOptions = ("F", "F'", "F2", "R", "R'", "R2", "U", "U'", "U2", "B", "B'", "B2", "L", "L'", "L2", "D", "D'", "D2")
    bad = True
    scramble = []
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
        match scrambleFaces[i]:
            case 0:
                move = moveOptions[randint(0, 2)]
                scramble.append(move)
            case 1:
                move = moveOptions[randint(3, 5)]
                scramble.append(move)
            case 2:
                move = moveOptions[randint(6, 8)]
                scramble.append(move)
            case 3:
                move = moveOptions[randint(9, 11)]
                scramble.append(move)
            case 4:
                move = moveOptions[randint(12, 14)]
                scramble.append(move)
            case 5:
                move = moveOptions[randint(13, 15)]
                scramble.append(move)

    return scramble


# Same as above, just with a custom length
def customScramble(length):

    faceOptions = (0, 1, 2, 3, 4, 5)
    moveOptions = ("F", "F'", "F2", "R", "R'", "R2", "U", "U'", "U2", "B", "B'", "B2", "L", "L'", "L2", "D", "D'", "D2")
    bad = True
    scramble = []
    scrambleFaces = []

    while bad is True:
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
        match scrambleFaces[i]:
            case 0:
                move = moveOptions[randint(0, 2)]
                scramble.append(move)
            case 1:
                move = moveOptions[randint(3, 5)]
                scramble.append(move)
            case 2:
                move = moveOptions[randint(6, 8)]
                scramble.append(move)
            case 3:
                move = moveOptions[randint(9, 11)]
                scramble.append(move)
            case 4:
                move = moveOptions[randint(12, 14)]
                scramble.append(move)
            case 5:
                move = moveOptions[randint(13, 15)]
                scramble.append(move)

    return scramble
