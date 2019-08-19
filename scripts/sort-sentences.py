import sys

if __name__ == '__main__':
  file = sys.argv[1]
  sents = {}
  with open(file, 'r') as fd:
     for _ in range(5):
       fd.readline()
     for line in fd:
       if line.startswith('|'):
         break
       elif line.startswith('S-'):
         id = line.split('\t')[0].split('-')[1]
       elif line.startswith('T-') or line.startswith('P-'):
         continue
       else:
         sents[id] = line

  for i in range(len(sents)):
    try:
      sys.stdout.write(sents[str(i)])
    except KeyError:
      continue
