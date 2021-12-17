import gen
from output import output

if __name__ == '__main__':
    # LC
    with open('inLC.txt') as lc_params:
        lines = lc_params.readlines()

    generator, n = gen.LC(*map(int, lines[:-1])), int(lines[-1])
    output(generator, n, 'outLC.txt')

    # MM
    with open('inMM.txt') as mm_params:
        lines = mm_params.readlines()

    lc1 = gen.LC(*map(int, lines[:4]))
    lc2 = gen.LC(*map(int, lines[4:8]))
    k, n = map(int, lines[8:])

    generator = gen.MM(lc1, lc2, k, n)
    output(generator, n, 'outMM.txt')
