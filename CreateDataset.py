import ScrambleGenerator as sg

Scramble_count = 10
move_count = 20


def main():
    d = open("Scrambles.txt", "w")
    for i in range(Scramble_count):
        d.write(sg.customScramble(move_count) + "\n")


if __name__ == '__main__':
    main()
    exit()
