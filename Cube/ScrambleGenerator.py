from random import randint

from pycuber import Formula, Cube
from pycuber.solver.cfop.cross import CrossSolver
from pycuber.solver.cfop.f2l import F2LSolver
from pycuber.solver.cfop.oll import OLLSolver
# from pycuber.solver.cfop.pll import PLLSolver


class Scramblegen:

    def __call__(self):
        pass

    # replaces match-case from python 3.10
    @staticmethod
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

    def FullScramble(self):
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
            scramble = self.moveswitcher(scrambleFaces[i]) + " " + scramble
        cube = Cube()
        return cube(Formula(scramble))

    # Same as above, just with a custom length
    def CustomScramble(self, length):

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
            scramble = self.moveswitcher(scrambleFaces[i]) + " " + scramble
        cube = Cube()
        return cube(Formula(scramble))

    # returns a cube with solved OLL
    def PLLScramble(self):
        scrambled = self.FullScramble()
        crss = CrossSolver(scrambled)
        crss.solve()
        cross = crss.cube

        f2ls = F2LSolver(cross)
        for _ in enumerate(f2ls.solve()):
            pass
        f2l = f2ls.cube

        olls = OLLSolver(f2l)
        olls.solve()
        return olls.cube

    # returns a cube with solved F2L
    def OLLScramble(self):
        scrambled = self.FullScramble()
        crss = CrossSolver(scrambled)
        crss.solve()
        cross = crss.cube

        f2ls = F2LSolver(cross)
        for _ in enumerate(f2ls.solve()):
            pass
        return f2ls.cube

    # returns a cube with a solved cross
    def F2LScramble(self):
        scrambled = self.FullScramble()
        solver = CrossSolver(scrambled)
        solver.solve()
        return solver.cube

# for test purposes only
# if __name__ == '__main__':
#     pll = Scramblegen()
#     pllscramble = pll.PLLScramble()
#     print(repr(pllscramble))
#     done = PLLSolver(pllscramble)
#     done.solve()
#     print(repr(done.cube))
