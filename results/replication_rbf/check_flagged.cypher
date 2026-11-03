// ---------------------------------------------------------------------------
// The 12 edges shown in panel 5 of explain_rbf_treats_rbf.pdf
// (the most anomalous biolink:treats edges under rbf / nu=0.2 / gamma=scale).
//
// Source of truth for these rows:
//   robokop_30fd_filtered/subgraph_edges.jsonl  ->  out/edges.tsv
// That is a FILTERED subgraph. Run these against ROBOKOP itself to confirm the
// edges are in the upstream graph and not an artifact of the filtering.
// ---------------------------------------------------------------------------

// --- 1. Do the edges exist?  One row per pair; found = false means absent. ---
UNWIND [
  {s:'UNII:F8H3J9KSY9',               o:'UMLS:C1136254'},  // Astodrimer -> Microbicides
  {s:'UNII:7SI2UZG934',               o:'UMLS:C2733598'},  // Condoliase -> Disorder of lumbar disc
  {s:'CHEMBL.COMPOUND:CHEMBL2108183', o:'UMLS:C3266799'},  // crilanomer -> Wound dressing
  {s:'UNII:WLT3PF2KX0',               o:'UMLS:C1960340'},  // air polymer-type A -> Fertiloscopy
  {s:'UNII:538316W9ZU',               o:'UMLS:C0034606'},  // Besilesomab -> Radionuclide Imaging
  {s:'CHEMBL.COMPOUND:CHEMBL1615807', o:'UMLS:C0203827'},  // indocyanine green -> Lymphatic imaging
  {s:'UNII:1N9O74Y20W',               o:'UMLS:C0034606'},  // NOFETUMOMAB -> Radionuclide Imaging
  {s:'UNII:S276568KOY',               o:'UMLS:C1512628'},  // Gadopiclenol -> imaging agent (product)
  {s:'PUBCHEM.COMPOUND:9952872',      o:'UMLS:C0563269'},  // zabofloxacin -> AECOPD
  {s:'UNII:X89XV46R07',               o:'UMLS:C0203668'},  // Tc 99m medronate -> Radioisotope bone scan
  {s:'CHEMBL.COMPOUND:CHEMBL1615807', o:'UMLS:C0412366'},  // indocyanine green -> vascular flow imaging
  {s:'UNII:Y68179SY2L',               o:'UMLS:C4707060'}   // Ga-68-Dotatoc -> PET/CT whole body
] AS pair
OPTIONAL MATCH (s {id: pair.s})-[r:`biolink:treats`]->(o {id: pair.o})
RETURN pair.s                        AS subject_id,
       s.name                        AS subject_name,
       labels(s)                     AS subject_labels,
       pair.o                        AS object_id,
       o.name                        AS object_name,
       labels(o)                     AS object_labels,
       r IS NOT NULL                 AS found,
       r.primary_knowledge_source    AS source,
       r.knowledge_level             AS knowledge_level,
       r.agent_type                  AS agent_type;


// --- 2. The type-rule view: which of them point at a non-disease object? -----
// A treats edge should run chemical -> disease/phenotype. Anything else is
// malformed by definition, independent of any model.
UNWIND ['UMLS:C1136254','UMLS:C2733598','UMLS:C3266799','UMLS:C1960340',
        'UMLS:C0034606','UMLS:C0203827','UMLS:C1512628','UMLS:C0563269',
        'UMLS:C0203668','UMLS:C0412366','UMLS:C4707060'] AS oid
MATCH (o {id: oid})
RETURN o.id    AS object_id,
       o.name  AS object_name,
       labels(o) AS object_labels,
       any(l IN labels(o) WHERE l IN ['biolink:Disease','biolink:PhenotypicFeature'])
              AS object_type_is_valid
ORDER BY object_type_is_valid, object_id;


// --- 3. Drill into one edge: everything ROBOKOP stores about it --------------
MATCH (s {id: 'UNII:538316W9ZU'})-[r:`biolink:treats`]->(o {id: 'UMLS:C0034606'})
RETURN properties(s) AS subject, type(r) AS predicate,
       properties(r) AS edge,    properties(o) AS object;


// --- 4. How many treats edges land on a Procedure across the whole graph? ----
// The population-level version of the panel-5 observation.
MATCH (s)-[r:`biolink:treats`]->(o:`biolink:Procedure`)
RETURN count(*) AS treats_edges_pointing_at_a_procedure;
