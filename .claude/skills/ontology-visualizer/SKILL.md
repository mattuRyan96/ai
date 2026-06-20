---
name: ontology-visualizer
description: Create an ontology (classes, relationships, individuals) as structured JSON and visualize it as an interactive graph using a Node.js server. Use when the user wants to model a domain/knowledge graph/taxonomy/ontology and see it rendered, or asks to "build an ontology", "make a knowledge graph", or "visualize concepts and relationships" with Node.
---

# Ontology Visualizer

Build an ontology as structured JSON, then render it as an interactive,
force-directed graph served by a small Node.js application.

## What "ontology" means here

An ontology is a formal model of a domain. This skill uses a pragmatic JSON
format (not full OWL/RDF, but mappable to it) with four parts:

- **classes** — the concepts/types in the domain, optionally arranged in a
  hierarchy via `subClassOf` (e.g. `Dog subClassOf Animal`).
- **relations** — named edge types between classes ("object properties"),
  each with a `domain` (source class) and `range` (target class)
  (e.g. `worksFor: Person -> Organization`).
- **individuals** — optional concrete instances of classes
  (e.g. `alice : Person`).
- **assertions** — optional facts linking individuals via relations
  (e.g. `alice worksFor acme`).

See `templates/ontology.schema.json` for the full JSON Schema and
`templates/example-ontology.json` for a worked example.

## Workflow

1. **Model the domain.** Interview the user (or infer from their description)
   for the key concepts, how they specialize each other, and how they relate.
   Write an `ontology.json` following the schema. Start small — classes and
   relations first; add individuals/assertions only if the user wants instance
   data. Validate as you go:
   - every `subClassOf` points to an existing class id
   - every relation `domain`/`range` references an existing class id
   - every individual `type` references an existing class id
   - every assertion `subject`/`object` references an existing individual id
     and `predicate` references an existing relation id

2. **Scaffold the visualizer.** Copy the files from `templates/` into the
   target project (default: a new `ontology-viz/` directory):
   - `server.js` — zero-dependency Node HTTP server. Serves the front-end and
     exposes the ontology at `/api/ontology` (raw) and `/api/graph`
     (transformed into nodes/edges for rendering).
   - `graph.js` — pure function that converts an ontology into a
     `{ nodes, edges }` graph. `subClassOf` and `type` become edges; relations
     become edge types; assertions become edges between individuals.
   - `public/index.html` — front-end that fetches `/api/graph` and renders it
     with [vis-network](https://visjs.github.io/vis-network/) (loaded from CDN).
   Place the `ontology.json` you built next to `server.js`.

3. **Run it.** No `npm install` is required — `server.js` uses only Node
   built-ins. Start with:
   ```bash
   node server.js            # serves http://localhost:3000
   node server.js ./my.json 8080   # custom ontology path + port
   ```
   Open the URL; the graph is interactive (drag, zoom, hover for definitions).

## Conventions & tips

- **Keep the model the source of truth.** All visual structure derives from
  `ontology.json`; never hand-edit the graph. Re-run after editing the JSON.
- **Color/shape by kind.** `graph.js` already tags nodes with `group`
  (`class` vs `individual`) and edges with `relation` (`subClassOf`, `type`,
  or a relation id) so the front-end can style them distinctly.
- **Offline use.** The front-end uses a vis-network CDN. If the user needs a
  fully offline build, vendor the library locally and update the `<script>`
  tag, or swap in `cytoscape` / `d3-force` — the `/api/graph` payload is a
  generic node-link format that any of these can consume.
- **Scaling to RDF/OWL.** The JSON maps cleanly to triples: classes →
  `rdfs:Class`, `subClassOf` → `rdfs:subClassOf`, relations →
  `owl:ObjectProperty` with `rdfs:domain`/`rdfs:range`, assertions → instance
  triples. Mention this if the user asks about Protégé / SPARQL / Turtle export.

## Validation script

After writing an ontology, sanity-check referential integrity by running the
transform — it throws on dangling references:
```bash
node -e "console.log(JSON.stringify(require('./graph').toGraph(require('./ontology.json')),null,2))"
```
