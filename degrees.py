"""Stage 2. Global node degree over the full graph.

Degree is needed twice downstream: as context on every flagged edge (so a flag
can be read as "wrong edge" rather than "rare node"), and as the degree-only
baseline in the evaluation. Both matter because degree is the failure mode that
invalidated an earlier node2vec run on the mf1 graph.

Uses the TSV copy of subgraph_edges.jsonl -- the same edges, minus the 1,262
documented dangling self-loops, which are irrelevant to a degree count.

Usage:  python degrees.py
"""
import collections

import config

config.require(config.DEG_SRC)
config.OUT.mkdir(parents=True, exist_ok=True)
config.refuse_overwrite(config.DEGREES_TSV)

indeg = collections.Counter()
outdeg = collections.Counter()
n = 0
with open(config.DEG_SRC) as fh:
    next(fh)
    for line in fh:
        s, _, o = line.rstrip("\n").split("\t")
        outdeg[s] += 1
        indeg[o] += 1
        n += 1

nodes = set(indeg) | set(outdeg)
with open(config.DEGREES_TSV, "w") as fh:
    fh.write("curie\tin_degree\tout_degree\n")
    for c in nodes:
        fh.write("%s\t%d\t%d\n" % (c, indeg[c], outdeg[c]))
print("edges", n, "nodes with any degree", len(nodes), "->", config.DEGREES_TSV)
