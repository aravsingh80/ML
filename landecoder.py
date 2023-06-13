s = "rdfef"
decDict = dict()
alph = "abcdefghijklmnopqrstuvwxyz"
for a in range(len(alph)): decDict[alph[len(alph)-1-a]] = alph[a]
out = ""
for s2 in s: 
    if s2 in decDict: out += decDict[s2]
    else: out += s2
print(out)