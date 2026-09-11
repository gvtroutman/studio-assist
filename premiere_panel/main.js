// main.js - the whole panel.
//
// Premiere Pro registers no COM automation, so the road into it is a CEP
// panel. This one is deliberately dumb: it listens on loopback and evaluates
// whatever ExtendScript studio_premiere_mcp.py posts to it, answering with the
// string ExtendScript returned. Every tool body, every helper and the JSON
// serializer live on the Python side, where the tests can see them; nothing
// here needs to change when a tool does.
//
//   GET  /      -> {bridge, app, version, port, requests}
//   POST /run   -> {"script": "..."}  ->  {"result": "<string>"}
//
// Only loopback is bound, and a request must carry Content-Type
// application/json - a browser page cannot send that cross-origin without a
// preflight nobody here answers, so a web page the user has open cannot drive
// their Premiere through this panel.

(function () {
  "use strict";

  var $status = document.getElementById("status");
  var $addr = document.getElementById("addr");
  var $reqs = document.getElementById("reqs");
  var $log = document.getElementById("log");
  function setStatus(text, klass) { $status.textContent = text; $status.className = klass || ""; }
  function log(msg, klass) {
    var d = document.createElement("div");
    d.className = klass || "";
    d.textContent = "[" + new Date().toISOString().substring(11, 19) + "] " + msg;
    $log.insertBefore(d, $log.firstChild);
    while ($log.childNodes.length > 60) $log.removeChild($log.lastChild);
  }

  // CEP puts its host API on window.__adobe_cep__; CSInterface.js is only a
  // wrapper around it, so the panel ships without that file.
  var cep = window.__adobe_cep__;
  if (!cep) {
    setStatus("cannot start - not running inside CEP", "err");
    return;
  }
  var http;
  try { http = require("http"); }
  catch (e) {
    setStatus("cannot start - Node is not enabled for this panel", "err");
    log("require('http') failed: " + e.message + ". The manifest must pass --enable-nodejs.", "err");
    return;
  }

  // The Python side reads the same variable, so setting it once in the user's
  // environment moves both ends.
  var port = parseInt(process.env.STUDIO_PREMIERE_PORT, 10) || 7787;
  var env = {};
  try { env = JSON.parse(cep.getHostEnvironment()); } catch (e) {}
  var requests = 0;

  function reply(res, code, obj) {
    var body = JSON.stringify(obj);
    res.writeHead(code, {"Content-Type": "application/json; charset=utf-8",
                         "Content-Length": Buffer.byteLength(body)});
    res.end(body);
  }

  var server = http.createServer(function (req, res) {
    var host = String(req.headers.host || "").split(":")[0];
    if (host !== "127.0.0.1" && host !== "localhost") return reply(res, 403, {error: "loopback only"});
    if (req.method === "GET") {
      return reply(res, 200, {bridge: "studio-premiere", app: env.appName || "PPRO",
                              version: env.appVersion || "?", port: port, requests: requests});
    }
    if (req.method !== "POST" || req.url !== "/run") return reply(res, 404, {error: "POST /run with {script}"});
    if (String(req.headers["content-type"] || "").indexOf("application/json") !== 0) {
      return reply(res, 415, {error: "Content-Type must be application/json"});
    }
    var chunks = [];
    req.on("data", function (c) { chunks.push(c); });
    req.on("end", function () {
      var script = null;
      try { script = JSON.parse(Buffer.concat(chunks).toString("utf8")).script; } catch (e) {}
      if (typeof script !== "string") return reply(res, 400, {error: "body must be JSON with a string 'script'"});
      requests += 1;
      $reqs.textContent = String(requests);
      try {
        // evalScript runs on Premiere's main thread and calls back when the
        // script returns; a modal dialog inside Premiere holds it, which is why
        // the Python side has a timeout and this side does not.
        cep.evalScript(script, function (out) { reply(res, 200, {result: String(out)}); });
      } catch (e) {
        log("evalScript threw: " + e.message, "err");
        reply(res, 500, {error: "evalScript threw: " + e.message});
      }
    });
  });
  server.on("error", function (e) {
    setStatus("cannot listen on 127.0.0.1:" + port + " - " + e.message, "err");
    log("Is another Premiere, or another panel, already using the port? Set STUDIO_PREMIERE_PORT to move both ends.", "err");
  });
  server.listen(port, "127.0.0.1", function () {
    setStatus("ready", "ok");
    $addr.textContent = "127.0.0.1:" + port;
    log("listening; " + (env.appName || "PPRO") + " " + (env.appVersion || ""));
  });
})();
