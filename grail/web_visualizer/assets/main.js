"use strict";

async function boot() {
  const res = await fetch("manifest.json", { cache: "no-store" });
  if (!res.ok) {
    document.getElementById("grid").textContent = "failed to load manifest.json";
    return;
  }
  const manifest = await res.json();
  document.getElementById("title").textContent = manifest.title || "GRAIL motion library";
  const summary = document.getElementById("summary");
  summary.textContent =
    `${manifest.num_motions} motions` +
    (manifest.num_with_video !== undefined
      ? ` · ${manifest.num_with_video} with video`
      : "");

  const grid = document.getElementById("grid");
  const tpl = document.getElementById("card-template");
  const cards = [];

  for (const m of manifest.motions) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.querySelector(".name").textContent = m.name;
    if (m.frames) {
      node.querySelector(".frames").textContent = `${m.frames} frames`;
    }
    const video = node.querySelector("video");
    if (m.video) {
      video.src = m.video;
      // Play/pause on hover — cheap preview, no autoplay across N cards.
      node.addEventListener("mouseenter", () => video.play().catch(() => {}));
      node.addEventListener("mouseleave", () => video.pause());
    } else {
      video.remove();
    }
    const dl = node.querySelector("dl.meta");
    if (m.meta && Object.keys(m.meta).length) {
      for (const [k, v] of Object.entries(m.meta)) {
        const dt = document.createElement("dt");
        dt.textContent = k;
        const dd = document.createElement("dd");
        dd.textContent = Array.isArray(v)
          ? v.map((x) => (typeof x === "number" ? x.toFixed(3) : x)).join(", ")
          : String(v);
        dl.append(dt, dd);
      }
    } else {
      dl.remove();
    }
    grid.append(node);
    cards.push({ name: m.name.toLowerCase(), node });
  }

  const search = document.getElementById("search");
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    for (const { name, node } of cards) {
      node.classList.toggle("hidden", !!q && !name.includes(q));
    }
  });
}

boot().catch((e) => {
  console.error(e);
  document.getElementById("grid").textContent = String(e);
});
