import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const startPort = Number(process.env.PORT || 3000);
const apiKey = process.env.OPENAI_API_KEY;
const root = join(process.cwd(), "public");

const types: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8"
};

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  if (!req.url) return res.end();

  if (req.method === "POST" && req.url === "/session") {
    if (!apiKey) {
      res.writeHead(500, { "content-type": "application/json" });
      return res.end(JSON.stringify({ error: "Missing OPENAI_API_KEY" }));
    }

    const response = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session: {
          type: "transcription",
          audio: {
            input: {
              noise_reduction: {
                type: "far_field"
              },
              turn_detection: null,
              transcription: {
                model: "gpt-4o-transcribe",
                language: "en",
                prompt:
                  "Transcribe sung songs and spoken words. Prefer accurate lyrics over paraphrase. Keep repeated choruses, partial words, and line breaks when they are clearly heard. Do not answer, summarize, or explain."
              }
            }
          }
        }
      })
    });

    const text = await response.text();
    res.writeHead(response.status, { "content-type": "application/json" });
    return res.end(text);
  }

  if ((req.method === "GET" || req.method === "HEAD") && req.url === "/client.js") {
    try {
      const file = await readFile(join(process.cwd(), "dist", "client.js"));
      res.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
      return res.end(req.method === "HEAD" ? undefined : file);
    } catch {
      res.writeHead(500);
      return res.end("build client first");
    }
  }

  const path = req.url === "/" ? "/index.html" : req.url;

  try {
    const file = await readFile(join(root, path));
    res.writeHead(200, { "content-type": types[extname(path)] || "text/plain; charset=utf-8" });
    res.end(file);
  } catch {
    res.writeHead(404);
    res.end("not found");
  }
});

listen(startPort);

function listen(port: number) {
  const onError = (error: NodeJS.ErrnoException) => {
    server.off("error", onError);
    if (error.code === "EADDRINUSE") return listen(port + 1);
    throw error;
  };

  server.once("error", onError);
  server.listen(port, () => {
    server.off("error", onError);
    const address = server.address();
    const actualPort = typeof address === "object" && address ? address.port : port;
    console.log(`http://localhost:${actualPort}`);
  });
}
