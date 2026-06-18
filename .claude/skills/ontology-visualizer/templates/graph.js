'use strict';

/**
 * Convert a pragmatic JSON ontology into a node-link graph
 * ({ nodes, edges }) suitable for vis-network / cytoscape / d3.
 *
 * Throws on dangling references so it doubles as a validator.
 */
function toGraph(onto) {
  if (!onto || !Array.isArray(onto.classes)) {
    throw new Error('Ontology must have a "classes" array.');
  }

  const classes = onto.classes;
  const relations = onto.relations || [];
  const individuals = onto.individuals || [];
  const assertions = onto.assertions || [];

  const classIds = new Set(classes.map((c) => c.id));
  const relationIds = new Set(relations.map((r) => r.id));
  const individualIds = new Set(individuals.map((i) => i.id));
  const relById = new Map(relations.map((r) => [r.id, r]));

  const must = (cond, msg) => { if (!cond) throw new Error(msg); };

  const nodes = [];
  const edges = [];

  // Classes -> nodes; subClassOf -> edges.
  for (const c of classes) {
    must(c.id, 'Every class needs an id.');
    nodes.push({
      id: c.id,
      label: c.label || c.id,
      group: 'class',
      title: c.comment || c.id
    });
    if (c.subClassOf != null) {
      must(classIds.has(c.subClassOf),
        `Class "${c.id}" subClassOf unknown class "${c.subClassOf}".`);
      edges.push({
        from: c.id,
        to: c.subClassOf,
        label: 'subClassOf',
        relation: 'subClassOf',
        arrows: 'to'
      });
    }
  }

  // Validate relation domain/range against known classes.
  for (const r of relations) {
    must(classIds.has(r.domain),
      `Relation "${r.id}" has unknown domain "${r.domain}".`);
    must(classIds.has(r.range),
      `Relation "${r.id}" has unknown range "${r.range}".`);
  }

  // Individuals -> nodes; type -> edge to their class.
  for (const ind of individuals) {
    must(classIds.has(ind.type),
      `Individual "${ind.id}" has unknown type "${ind.type}".`);
    nodes.push({
      id: ind.id,
      label: ind.label || ind.id,
      group: 'individual',
      title: `instance of ${ind.type}`
    });
    edges.push({
      from: ind.id,
      to: ind.type,
      label: 'type',
      relation: 'type',
      arrows: 'to',
      dashes: true
    });
  }

  // Assertions -> edges between individuals.
  for (const a of assertions) {
    must(individualIds.has(a.subject),
      `Assertion subject "${a.subject}" is not a known individual.`);
    must(individualIds.has(a.object),
      `Assertion object "${a.object}" is not a known individual.`);
    must(relationIds.has(a.predicate),
      `Assertion predicate "${a.predicate}" is not a known relation.`);
    const rel = relById.get(a.predicate);
    edges.push({
      from: a.subject,
      to: a.object,
      label: rel.label || rel.id,
      relation: rel.id,
      arrows: 'to'
    });
  }

  return { nodes, edges };
}

module.exports = { toGraph };
