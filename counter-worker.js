// View counter worker for songqikong.github.io
//
// Counts page views per path (explicitly passed in the URL, NOT derived from
// the Referer header — so it works in every browser, including Safari /
// strict-privacy modes that strip cross-site referrers).
//
// API:
//   GET /count?path=/2023-11-02-hdr/   -> increments and returns the new count
//   {"path":"/2023-11-02-hdr/","count":12}
//
// Deployment (Cloudflare dashboard, free plan is enough):
//   1. Workers & Pages -> Create Worker -> name it (e.g. "view-counter")
//   2. Replace the default code with this file's contents -> Deploy
//   3. Workers & Pages -> KV -> Create a namespace (e.g. "VIEWS_KV")
//   4. In the worker's Settings -> Variables -> KV namespace bindings:
//      Variable name: VIEWS_KV, KV namespace: the one from step 3
//   5. Re-deploy the worker if the binding was added after deploy
//   6. Copy the worker URL (e.g. https://view-counter.<subdomain>.workers.dev)
//      into `view_counter_url` in _config.yml and push.
//
// Note: the counter is public — anyone who knows the worker URL can increment
// any path (same trust model as busuanzi).

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: CORS });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    const url = new URL(request.url);
    if (url.pathname !== "/count") return json({ error: "not found" }, 404);

    const path = (url.searchParams.get("path") || "").slice(0, 200);
    if (!path) return json({ error: "missing path" }, 400);

    const key = "pv:" + path;
    const current = parseInt((await env.VIEWS_KV.get(key)) || "0", 10);
    const next = (Number.isFinite(current) ? current : 0) + 1;
    await env.VIEWS_KV.put(key, String(next));

    return json({ path, count: next });
  },
};
