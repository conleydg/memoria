// Shared helpers for the model lab pages. No external libraries.

const PAGES = [["/", "Gallery"], ["/ask", "Ask"], ["/search", "Search playground"], ["/map", "Embedding map"], ["/compare", "Compare models"], ["/stats", "Model stats"], ["/mlx", "MLX hands-on"]];

function header(active) {
  const nav = PAGES.map(([href, name]) => `<a href="${href}" class="${href === active ? "on" : ""}">${name}</a>`).join("");
  document.body.insertAdjacentHTML("afterbegin",
    `<header class="top"><span class="brand">memoria · model lab</span><nav>${nav}</nav>
     <span class="priv">local only · 127.0.0.1 · nothing leaves this Mac</span></header>`);
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const fmtDate = (ts) => (ts ? new Date(ts * 1000).toISOString().slice(0, 10) : "–");
const fmtBytes = (b) => (b == null ? "–" : b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB");
function fmtDuration(s) {
  if (s == null) return "–";
  if (s < 120) return s.toFixed(0) + " s";
  if (s < 7200) return (s / 60).toFixed(0) + " min";
  if (s < 172800) return (s / 3600).toFixed(1) + " h";
  return (s / 86400).toFixed(1) + " days";
}

function card(a, opts = {}) {
  if (!a) return "";
  const score = opts.score != null ? `<span class="score">${opts.score}</span>` : "";
  const vid = a.kind === "video" ? `<span class="badge video">▶ ${a.duration ? a.duration.toFixed(0) + "s" : "video"}</span>` : "";
  return `<a class="card ${opts.dim ? "dim" : ""}" href="/asset?id=${a.uuid}" title="${esc(opts.title || "")}">
    <img loading="lazy" src="/media/${a.uuid}/thumb.jpg" alt="">${vid}${score}
    <div class="meta"><span>${a.year ?? ""}</span><span>${esc(a.label ?? "")}</span>
    <span>${a.aesthetic != null ? "★ " + fmt(a.aesthetic, 1) : ""}</span></div>
    ${a.filename ? `<div class="fname mono" title="Click to copy. Search for it in Photos." data-copy="${esc(a.filename)}">${esc(a.filename)}</div>` : ""}</a>`;
}

function explainer(title, html, open = false) {
  return `<details class="explain" ${open ? "open" : ""}><summary>What am I looking at? · ${esc(title)}</summary>${html}</details>`;
}

const LABEL_COLORS = { photo: "#4c8bf5", screenshot: "#e8710a", document: "#9334e6", receipt: "#1e8e3e" };
function yearColor(y, min, max) {
  const t = max > min ? (y - min) / (max - min) : 0.5;
  return `hsl(${Math.round(260 - 220 * t)}, 70%, 50%)`;
}

// Click a filename to copy it (to paste into Photos' search), without opening the card.
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-copy]");
  if (!el) return;
  e.preventDefault(); e.stopPropagation();
  const text = el.dataset.copy, old = el.textContent;
  const done = () => { el.textContent = "copied ✓"; setTimeout(() => (el.textContent = old), 1200); };
  (navigator.clipboard?.writeText(text) ?? Promise.reject()).then(done).catch(() => {
    const t = Object.assign(document.createElement("textarea"), { value: text });
    document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); done();
  });
}, true);
