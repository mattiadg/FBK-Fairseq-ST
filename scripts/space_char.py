# sed -i 's/original/new/g' file.txt

import sys

with open(sys.argv[1]) as f:
	l = f.readlines()

with open(sys.argv[2], 'a') as g:
	for line in l:
		g.write(" ".join(line))




