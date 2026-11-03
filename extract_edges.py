"""Stage 1. One pass over subgraph_edges.jsonl: pull the chemical-disease
relations of interest, and take a full predicate census while we're reading
anyway.

Only two of the extracted predicates are used downstream -- biolink:treats as
the positives the detector is fit on, and biolink:contraindicated_in as the
held-out negatives it is evaluated against. The rest are extracted because the
pass is the expensive part, and having them makes a different negative choice a
one-flag change rather than a re-read.

Usage:  python extract_edges.py
"""
import collections
import json

import config

WANTED = {
    "biolink:treats",                                 # positives
    "biolink:contraindicated_in",                     # evaluation negatives
    "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:ameliorates_condition",
    "biolink:preventative_for_condition",
    "biolink:has_adverse_event",
}
# cheap substring prefilter so we only json-parse candidate lines
HINTS = ("treat", "ameliorat", "preventative", "contraindicat", "adverse")

QUAL_KEYS = (
    "subject_aspect_qualifier", "subject_direction_qualifier",
    "object_aspect_qualifier", "object_direction_qualifier",
    "qualified_predicate",
)

config.require(config.EDGES_JSONL)
config.OUT.mkdir(parents=True, exist_ok=True)
config.refuse_overwrite(config.EDGES_TSV)

census = collections.Counter()
qual_seen = collections.Counter()
kept = 0

with open(config.EDGES_TSV, "w") as fh:
    fh.write("subject\tpredicate\tobject\tprimary_knowledge_source\tqualifiers\n")
    for line in open(config.EDGES_JSONL):
        if not any(h in line for h in HINTS):
            continue
        d = json.loads(line)
        p = d["predicate"]
        census[p] += 1
        if p not in WANTED:
            continue
        quals = {k: d[k] for k in QUAL_KEYS if d.get(k)}
        if quals:
            qual_seen[(p, tuple(sorted(quals)))] += 1
        fh.write("{}\t{}\t{}\t{}\t{}\n".format(
            d["subject"], p, d["object"],
            d.get("primary_knowledge_source", ""),
            json.dumps(quals, separators=(",", ":")) if quals else ""))
        kept += 1

with open(config.OUT / "census.txt", "w") as fh:
    fh.write("kept\t%d\n\n# predicate census among prefilter-matching lines\n" % kept)
    for p, n in census.most_common():
        fh.write("%s\t%d\n" % (p, n))
    fh.write("\n# qualified rows among WANTED predicates (expected: none)\n")
    for (p, ks), n in qual_seen.most_common():
        fh.write("%s\t%s\t%d\n" % (p, ",".join(ks), n))

print("kept", kept, "edges ->", config.EDGES_TSV)
print("census:", dict(census.most_common(12)))
print("qualified WANTED rows:", dict(qual_seen) or "none")
