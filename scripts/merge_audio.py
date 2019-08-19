import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--input', required=True, nargs='+', help="List of file paths to merge")
args.add_argument('--output', required=True, help="Path of the output file")

if __name__ == '__main__':
    opts = args.parse_args()
    sizes = (0, 0)
    for file in opts.input:
        with open(file, 'rb') as f:
            sz = np.load(f)
            if sizes[0] == sizes[1] == 0:
                sizes = sz
            else:
                if sizes[1] == sz[1]:
                    sizes = sizes[0] + sz[0], sz[1]
                else:
                    raise ValueError("All the files must have the same number of features!")

    with open(opts.output, 'wb') as f_out:
        np.save(f_out, np.array(list(sizes)))
        for file in opts.input:
            print(file)
            with open(file, 'rb') as f:
                rows, _ = np.load(f)
                for i in range(rows):
                    x = np.load(f)
                    np.save(f_out, x)
