'use strict';

/**
 * Zero-dependency Node.js server for the ontology visualizer.
 * Uses only Node built-ins — no `npm install` required.
 *
 *   node server.js                       # ./ontology.json on :3000
 *   node server.js ./my-ontology.json    # custom ontology
 *   node server.js ./my-ontology.json 8080
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { toGraph } = require('./graph');

const ONTOLOGY_PATH = path.resolve(process.argv[2] || './ontology.json');
const PORT = Number(process.argv[3]) || 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8'
};

function readOntology() {
  const raw = fs.readFileSync(ONTOLOGY_PATH, 'utf8');
  return JSON.parse(raw);
}

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj, null, 2);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(body);
}

function serveStatic(req, res) {
  // Map "/" to index.html; prevent path traversal.
  const rel = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]);
  const filePath = path.join(PUBLIC_DIR, path.normalize(rel));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  try {
    if (req.url === '/api/ontology') {
      return sendJson(res, 200, readOntology());
    }
    if (req.url === '/api/graph') {
      return sendJson(res, 200, toGraph(readOntology()));
    }
    return serveStatic(req, res);
  } catch (err) {
    return sendJson(res, 500, { error: err.message });
  }
});

server.listen(PORT, () => {
  console.log(`Ontology:   ${ONTOLOGY_PATH}`);
  console.log(`Visualizer: http://localhost:${PORT}`);
});
