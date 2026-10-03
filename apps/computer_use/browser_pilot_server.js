// Local test fixture: ordinary form pages, no answer key or state-read API.
'use strict';
const fs = require('fs');
const http = require('http');
const path = require('path');
const [directory, portText] = process.argv.slice(2);
const port = Number(portText);
if (!directory || !Number.isInteger(port) || port < 1024 || port > 65535) throw Error('Bad fixture arguments');
const log = path.join(directory, 'submissions.jsonl');
const keys = ['name', 'email', 'topic', 'delivery', 'notes', 'consent'];
const escape = value => String(value || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const document = body => '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QW3 Local Request</title><style>body{font:18px system-ui;max-width:660px;margin:24px auto;padding:0 18px}label{display:block;margin:18px 0 5px}input:not([type=checkbox]),select,textarea{box-sizing:border-box;width:100%;font:inherit;padding:9px}button{font:inherit;padding:12px 20px;margin:20px 12px 10px 0}dt{font-weight:bold;margin-top:12px}dd{margin:4px 0;white-space:pre-wrap}</style></head><body><h1>Local support request</h1><p>This is an offline practice form. Review the details before saving.</p>' + body + '</body></html>';
function form(values = {}) {
  const option = (field, value, label) => '<option value="' + value + '"' + (values[field] === value ? ' selected' : '') + '>' + label + '</option>';
  return document('<form method="post" action="/review">' +
    '<label for="name">Full name</label><input id="name" name="name" required maxlength="100" value="' + escape(values.name) + '">' +
    '<label for="email">Email address</label><input id="email" name="email" type="email" required maxlength="150" value="' + escape(values.email) + '">' +
    '<label for="topic">Topic</label><select id="topic" name="topic" required><option value="">Choose a topic</option>' + option('topic','python','Python environment') + option('topic','latex','LaTeX installation') + option('topic','desktop','Desktop assistance') + '</select>' +
    '<label for="delivery">Preferred delivery</label><select id="delivery" name="delivery">' + option('delivery','email','Email') + option('delivery','local','Local notification') + '</select>' +
    '<label for="notes">Details</label><textarea id="notes" name="notes" required rows="3" maxlength="500">' + escape(values.notes) + '</textarea>' +
    '<label><input name="consent" type="checkbox" value="yes" required' + (values.consent === 'yes' ? ' checked' : '') + '> I confirm these are test details</label><button type="submit">Review request</button></form>');
}
function review(values) {
  const summary = keys.map(key => '<dt>' + escape(key) + '</dt><dd>' + escape(values[key]) + '</dd>').join('');
  const fields = keys.map(key => '<input type="hidden" name="' + key + '" value="' + escape(values[key]) + '">').join('');
  return document('<h2>Review before saving</h2><dl>' + summary + '</dl><form method="post" action="/submit">' + fields + '<button type="submit">Save request</button><button type="submit" formaction="/edit">Edit details</button></form>');
}
function parse(body) {
  const params = new URLSearchParams(body);
  if ([...params.keys()].some(key => !keys.includes(key)) || keys.some(key => params.getAll(key).length !== 1)) throw Error('Missing, duplicate or unsupported fields');
  const values = Object.fromEntries(keys.map(key => [key, params.get(key)]));
  if (!values.name.trim() || values.name.length > 100 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email) || values.email.length > 150 ||
      !['python','latex','desktop'].includes(values.topic) || !['email','local'].includes(values.delivery) ||
      !values.notes.trim() || values.notes.length > 500 || values.consent !== 'yes') throw Error('Invalid form fields');
  return values;
}
const server = http.createServer((req, res) => {
  const send = (status, body) => { res.writeHead(status, {'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'}); res.end(body); };
  if (req.method === 'GET' && req.url === '/') return send(200, form());
  if (req.method !== 'POST' || !['/review','/edit','/submit'].includes(req.url)) return send(404, document('<p>Page not found.</p>'));
  if (!(req.headers['content-type'] || '').startsWith('application/x-www-form-urlencoded')) return send(415, document('<p>Use the form to continue.</p>'));
  let chunks = [], bytes = 0, oversized = false;
  req.on('data', data => { bytes += data.length; if (bytes > 8192) oversized = true; else chunks.push(data); });
  req.on('end', () => {
    if (oversized) return send(413, document('<p>Request too large.</p>'));
    let values;
    try { values = parse(Buffer.concat(chunks).toString('utf8')); } catch (error) { return send(400, document('<p>Check all required fields and return to the form.</p>')); }
    if (req.url === '/review') return send(200, review(values));
    if (req.url === '/edit') return send(200, form(values));
    const fd = fs.openSync(log, 'a', 0o600);
    try { fs.writeSync(fd, JSON.stringify({submitted_at:Date.now()/1000, values}) + '\n'); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
    return send(200, document('<h2>Request saved</h2><p>Your test request is saved on this device.</p><dl>' + keys.map(key => '<dt>' + escape(key) + '</dt><dd>' + escape(values[key]) + '</dd>').join('') + '</dl>'));
  });
});
server.headersTimeout = 10000;
server.requestTimeout = 15000;
server.listen(port, '127.0.0.1', () => {
  fs.writeFileSync(path.join(directory, 'server-ready.json'), JSON.stringify({pid:process.pid, port, bind:'127.0.0.1', source:__filename}));
  process.stdout.write('Local form fixture ready\n');
});
process.on('SIGTERM', () => server.close(() => process.exit(0)));
