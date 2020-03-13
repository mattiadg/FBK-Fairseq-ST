import sys

if __name__ == '__main__':
  file = sys.argv[1]
  offset = int(sys.argv[2])
  sents = {}
  with open(file, 'r') as fd:
     for _ in range(offset):
       fd.readline()
     for line in fd:
       if line.startswith('|'):
         continue
       elif line.startswith('S-') or line.startswith('T-') or line.startswith('P-'):
         continue
       elif line.startswith('H-'):
         id = line.split('\t')[0].split('-')[1]
         sents[id] = line.split('\t')[2]

  for i in range(len(sents)):
    try:
      sys.stdout.write(sents[str(i)])
    except KeyError:
      print("Error in", i)
      continue
