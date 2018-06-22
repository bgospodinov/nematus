import os
import sys
import random
import tempfile
import logging


def main(files, temporary=False, seed=None):
    if seed:
        logging.info("Setting random seed to {}".format(seed))
        random.seed(seed)
    fds = [open(ff, encoding='utf-8') for ff in files]

    lines = []
    for l in fds[0]:
        line = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        lines.append(line)

    [ff.close() for ff in fds]

    random.shuffle(lines)

    if temporary:
        fds = []
        for ff in files:
            path, filename = os.path.split(os.path.realpath(ff))
            fds.append(tempfile.TemporaryFile(mode="w+", prefix=filename+'.shuf', encoding='utf-8', dir=path))
    else:
        fds = [open(ff+'.shuf','w', encoding='utf-8') for ff in files]

    for l in lines:
        for ii, fd in enumerate(fds):
            print(l[ii], file=fd)

    if temporary:
        [ff.seek(0) for ff in fds]
    else:
        [ff.close() for ff in fds]

    return fds

if __name__ == '__main__':
    main(sys.argv[1:])

    


