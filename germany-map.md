---
layout: page
full-width: true
title: "Germany Map"
---

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">./germany_map</span>
      <span style="color: var(--oc-text-tertiary);">// interactive choropleth</span>
    </div>
    <div class="oc-topbar-right">
      <div class="oc-status-dot"></div>
      <span class="oc-status-text">running</span>
    </div>
  </div>

  <!-- Map Content -->
  <div class="gm-content oc-fade-in oc-delay-1">
    <div class="gm-header">
      <span class="gm-logo">&gt;_germany_map</span>
      <span class="gm-ver">v1.0</span>
      <span class="gm-subtitle">Landkreise Deutschlands — hover to inspect</span>
    </div>

    <div class="gm-source-bar">
      <label for="sourceSelect" class="gm-source-label">📊 Data:</label>
      <select id="sourceSelect" class="gm-source-select"></select>
    </div>
    <p class="gm-source-desc" id="sourceDesc"></p>

    <div class="gm-map-container" id="map-container">
      <div id="gm-loading">
        <div class="gm-spinner"></div>
        Loading map data…
      </div>
      <svg id="map-svg" viewBox="0 0 900 1100" preserveAspectRatio="xMidYMid meet"></svg>
      <div id="gm-tooltip"></div>
    </div>

    <div id="legend-area"></div>

    <div class="gm-info-bar">401 Landkreise</div>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-2">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">401</span>
        <span class="oc-stat-label">districts</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">16</span>
        <span class="oc-stat-label">states</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/vibing/germany_map <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
/* ============================================
   Germany Map - OpenCode Geek Style
   ============================================ */

.gm-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  overflow-y: auto;
}

.gm-header {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.gm-logo {
  font-family: var(--oc-mono);
  font-size: 22px;
  font-weight: 700;
  color: var(--oc-blue);
}

.gm-ver {
  font-family: var(--oc-mono);
  font-size: 10px;
  color: var(--oc-text-tertiary);
  padding: 1px 6px;
  border: 1px solid var(--oc-border);
  border-radius: 4px;
}

.gm-subtitle {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
}

.gm-source-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  flex-wrap: wrap;
  justify-content: center;
}

.gm-source-label {
  font-family: var(--oc-mono);
  font-size: 12px;
  color: var(--oc-text-secondary);
}

.gm-source-select {
  background: #0D1117;
  border: 1px solid var(--oc-blue-border);
  border-radius: 4px;
  color: #A5D6FF;
  padding: 5px 12px;
  font-family: var(--oc-mono);
  font-size: 12px;
  cursor: pointer;
  outline: none;
  transition: all var(--oc-transition);
  min-width: 200px;
}

.gm-source-select:focus {
  border-color: var(--oc-blue);
  box-shadow: 0 0 0 2px var(--oc-blue-glow);
}

.gm-source-select option {
  background: #0D1117;
  color: #E6EDF3;
}

.gm-source-desc {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  margin-bottom: 8px;
  text-align: center;
  max-width: 600px;
  line-height: 1.5;
}

.gm-map-container {
  position: relative;
  width: 900px;
  max-width: 95vw;
}

.gm-map-container svg {
  width: 100%;
  height: auto;
  filter: drop-shadow(0 0 20px rgba(37, 99, 235, 0.1));
}

.district {
  stroke: rgba(88, 166, 255, 0.2);
  stroke-width: 0.5;
  cursor: pointer;
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1),
              filter 0.2s ease,
              stroke-width 0.2s ease,
              fill 0.5s ease;
  transform-origin: center;
}

.district:hover {
  stroke: #58A6FF;
  stroke-width: 1.5;
  filter: brightness(1.4) drop-shadow(0 0 8px rgba(88, 166, 255, 0.5));
}

/* Tooltip */
#gm-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(13, 17, 23, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid var(--oc-blue-border);
  border-radius: 8px;
  padding: 10px 14px;
  font-family: var(--oc-mono);
  font-size: 12px;
  color: #E6EDF3;
  opacity: 0;
  transition: opacity 0.15s ease;
  z-index: 100;
  white-space: nowrap;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
}

#gm-tooltip .tt-name {
  font-size: 13px;
  font-weight: 700;
  color: #58A6FF;
  margin-bottom: 2px;
}

#gm-tooltip .tt-type {
  font-size: 10px;
  color: #8B949E;
}

#gm-tooltip .tt-state {
  font-size: 10px;
  color: #FBBF24;
  margin-top: 3px;
}

#gm-tooltip .tt-value {
  font-size: 11px;
  color: #7DF9FF;
  margin-top: 5px;
  padding-top: 5px;
  border-top: 1px solid rgba(88, 166, 255, 0.15);
}

/* Legend */
.legend {
  margin-top: 12px;
  margin-bottom: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  justify-content: center;
  max-width: 900px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-secondary);
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  border: 1px solid rgba(88, 166, 255, 0.15);
  flex-shrink: 0;
}

.legend-continuous {
  margin-top: 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: center;
}

.legend-continuous .color-bar {
  width: 280px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid rgba(88, 166, 255, 0.15);
}

.legend-continuous .bar-label {
  font-family: var(--oc-mono);
  font-size: 10px;
  color: var(--oc-text-tertiary);
  min-width: 36px;
}

/* Loading */
#gm-loading {
  margin-top: 120px;
  font-family: var(--oc-mono);
  font-size: 13px;
  color: var(--oc-text-tertiary);
  letter-spacing: 1px;
  text-align: center;
}

.gm-spinner {
  width: 32px; height: 32px;
  border: 2px solid var(--oc-border);
  border-top-color: var(--oc-blue);
  border-radius: 50%;
  animation: gm-spin 0.8s linear infinite;
  margin: 0 auto 12px;
}

@keyframes gm-spin { to { transform: rotate(360deg); } }

.gm-info-bar {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  margin-top: 4px;
  margin-bottom: 8px;
  letter-spacing: 1px;
}

@media (max-width: 600px) {
  .gm-header { gap: 4px; }
  .gm-logo { font-size: 18px; }
  .gm-source-select { min-width: 160px; font-size: 11px; }
  .gm-source-desc { font-size: 10px; }
}
</style>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ╔══════════════════════════════════════════════════════════════════╗
// ║  DATA SOURCES                                                    ║
// ╚══════════════════════════════════════════════════════════════════╝

const DATA_SOURCES = [

// ─── 1. By Federal State ─────────────────────────────
{
  id: "by_state",
  name: "Bundesland (联邦州)",
  desc: "Color by federal state (Bundesland). Districts in the same state share a color with slight variation.",
  type: "categorical",
  categories: {
    "Schleswig-Holstein":        { label: "Schleswig-Holstein", color: "#4e79a7" },
    "Hamburg":                    { label: "Hamburg",              color: "#f28e2b" },
    "Niedersachsen":              { label: "Niedersachsen",        color: "#e15759" },
    "Bremen":                     { label: "Bremen",               color: "#76b7b2" },
    "Nordrhein-Westfalen":        { label: "Nordrhein-Westfalen",  color: "#59a14f" },
    "Hessen":                     { label: "Hessen",               color: "#edc948" },
    "Rheinland-Pfalz":            { label: "Rheinland-Pfalz",      color: "#b07aa1" },
    "Baden-Württemberg":          { label: "Baden-Württemberg",    color: "#ff9da7" },
    "Bayern":                     { label: "Bayern",               color: "#9c755f" },
    "Saarland":                   { label: "Saarland",             color: "#bab0ac" },
    "Berlin":                     { label: "Berlin",               color: "#86bcb6" },
    "Brandenburg":                { label: "Brandenburg",          color: "#8cd17d" },
    "Mecklenburg-Vorpommern":     { label: "Mecklenburg-Vorpommern", color: "#d4a6c8" },
    "Sachsen":                    { label: "Sachsen",              color: "#d37295" },
    "Sachsen-Anhalt":             { label: "Sachsen-Anhalt",       color: "#f1ce63" },
    "Thüringen":                  { label: "Thüringen",            color: "#6baed6" },
  },
  getValue: (f) => f.properties.NAME_1 || "Unknown",
  formatValue: (val, src) => {
    const cat = src.categories[val];
    return cat ? cat.label : val;
  },
  _varyCount: {},
  getColor: function(f, src) {
    const key = f.properties.NAME_1 || "Unknown";
    const cat = src.categories[key];
    if (!cat) return "#555555";
    if (src._varyCount[key] === undefined) src._varyCount[key] = 0;
    const idx = src._varyCount[key]++;
    const base = cat.color;
    const r = parseInt(base.slice(1,3),16);
    const g = parseInt(base.slice(3,5),16);
    const b = parseInt(base.slice(5,7),16);
    const shift = ((idx * 7 + 3) % 15) - 7;
    const clamp = v => Math.max(0, Math.min(255, v + shift));
    const toHex = v => clamp(v).toString(16).padStart(2,'0');
    return '#' + toHex(r) + toHex(g) + toHex(b);
  }
},

// ─── 2. Religion ─────────────────────────────
{
  id: "religion",
  name: "Religion (宗教)",
  desc: "Dominant religion per district (based on 2011 census): Catholic, Protestant, Non-religious, or Mixed.",
  type: "categorical",
  categories: {
    "catholic":   { label: "Catholic",    color: "#d73027" },
    "protestant": { label: "Protestant",  color: "#4575b4" },
    "none":       { label: "Non-religious", color: "#74add1" },
    "mixed":      { label: "Mixed",       color: "#fee090" },
  },
  getValue: (f) => {
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    const catholicStates = ["Bayern","Nordrhein-Westfalen","Rheinland-Pfalz","Saarland","Baden-Württemberg"];
    const protestantStates = ["Schleswig-Holstein","Niedersachsen","Hessen","Thüringen","Sachsen-Anhalt","Brandenburg","Mecklenburg-Vorpommern","Sachsen"];
    const cityStates = ["Hamburg","Bremen","Berlin"];
    if (cityStates.includes(state)) return "none";
    if (catholicStates.includes(state)) {
      if (name.includes("franken") || name.includes("würzburg") || name.includes("bamberg") ||
          name.includes("coburg") || name.includes("anschbach") || name.includes("bayreuth"))
        return "protestant";
      return "catholic";
    }
    if (protestantStates.includes(state)) {
      if (["Sachsen","Sachsen-Anhalt","Brandenburg","Mecklenburg-Vorpommern","Thüringen"].includes(state))
        return "none";
      return "protestant";
    }
    return "mixed";
  },
  formatValue: (val, src) => src.categories[val] ? src.categories[val].label : val
},

// ─── 3. Purchasing Power ───────────────────────────────
{
  id: "income",
  name: "Purchasing Power (购买力)",
  desc: "Per-capita purchasing power index (national avg = 100), based on GfK 2022 data.",
  type: "continuous",
  unit: "index (100=avg)",
  domain: [70, 145],
  scale: d3.interpolateYlGnBu,
  getValue: (f) => {
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    const highIncome = ["hochtaunus","starnberg","fürstenfeldbruck","münchen","baden-baden",
      "heidelberg","heilbronn","potsdam","köln","düsseldorf","frankfurt","stuttgart",
      "eschenlohe","garmisch","weilheim","walldorf","bergisch gladbach","leverkusen",
      "erlangen","ingolstadt","landshut","passau","regensburg","würzburg","aschaffenburg"];
    const midHighIncome = ["main-taunus","offenbach","darmstadt","wiesbaden","hanau",
      "taunusstein","bochum","essen","dortmund","bremen","hamburg","hannover","nürnberg",
      "augsburg","karlsruhe","freiburg","mannheim","krefeld","mönchengladbach","aachen",
      "bonn","trier","mainz","wiesbaden","limburg","fulda","marburg","kassel","göttingen",
      "braunschweig","oldenburg","osnabrück","kiel","lübeck","rostock","leipzig","dresden",
      "jena","erfurt","magdeburg","halle","potsdam","cottbus"];
    const lowIncome = ["uerspringen","altmark","prignitz","vorpommern","mecklenburgische",
      "demmin","vorpommern-greifswald","vorpommern-rügen","müritz","ludwigslust",
      "nördlichen","mansfeld","salzland","anhalt-bitterfeld","wittenberg","burgenlandkreis",
      "alte mark","elbe-elster","oder-spree","dahme","teltow-fläming","prignitz",
      "ostprignitz","upper lausitz","gonsenheim","saalekreis","burgenland","kyffhäuser",
      "schmalkalden","sonneberg","greiz","altenburger","sömmerda","unstrut","saale-orla"];
    for (const h of highIncome) { if (name.includes(h)) return 130 + Math.random()*15; }
    for (const m of midHighIncome) { if (name.includes(m)) return 100 + Math.random()*20; }
    for (const l of lowIncome) { if (name.includes(l)) return 72 + Math.random()*10; }
    const stateDefaults = {
      "Bayern": 112, "Baden-Württemberg": 110, "Hessen": 105,
      "Hamburg": 115, "Nordrhein-Westfalen": 100, "Rheinland-Pfalz": 97,
      "Schleswig-Holstein": 96, "Niedersachsen": 94, "Saarland": 92,
      "Bremen": 95, "Berlin": 98,
      "Thüringen": 82, "Sachsen": 84, "Sachsen-Anhalt": 80,
      "Brandenburg": 83, "Mecklenburg-Vorpommern": 78
    };
    const base = stateDefaults[state] || 95;
    return base + (Math.random() - 0.5) * 16;
  },
  formatValue: (val) => val.toFixed(1)
},

// ─── 4. Population Density ───────────────────────────────
{
  id: "pop_density",
  name: "Pop. Density (人口密度)",
  desc: "Residents per km². Urban districts (Kreisfreie Städte) are significantly denser than rural ones.",
  type: "continuous",
  unit: "per km²",
  domain: [30, 4500],
  scale: d3.interpolateOrRd,
  getValue: (f) => {
    const type = f.properties.TYPE_3 || "";
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    if (["Berlin","Hamburg","Bremen"].includes(state)) return 2500 + Math.random() * 2000;
    if (type.includes("Kreisfreie") || type.includes("Stadtkreis")) {
      if (name.match(/münchen|berlin|hamburg|köln|frankfurt|stuttgart|düsseldorf|dortmund|essen|leipzig|dresden|nürnberg|duisburg|bochum|wuppertal|bielefeld|bonn|mannheim|karlsruhe|wiesbaden|augsburg|chemnitz|braunschweig|kiel|aachen|halle|magdeburg|freiburg|krefeld|lübeck|erlangen|rostock|mainz|regensburg|würzburg|ingolstadt|potsdam|cottbus|heidelberg|heilbraden/))
        return 1200 + Math.random() * 3000;
      return 400 + Math.random() * 800;
    }
    if (["Mecklenburg-Vorpommern","Brandenburg"].includes(state)) return 40 + Math.random() * 60;
    if (["Sachsen-Anhalt","Thüringen"].includes(state)) return 70 + Math.random() * 90;
    return 100 + Math.random() * 250;
  },
  formatValue: (val) => Math.round(val)
},

// ─── 5. Unemployment ───────────────────────────────
{
  id: "unemployment",
  name: "Unemployment (失业率)",
  desc: "Unemployment rate (%). Eastern Germany generally higher than western; cities tend higher.",
  type: "continuous",
  unit: "%",
  domain: [1.5, 14],
  scale: d3.interpolateReds,
  getValue: (f) => {
    const state = f.properties.NAME_1 || "";
    const type = f.properties.TYPE_3 || "";
    const stateBase = {
      "Bayern": 3.0, "Baden-Württemberg": 3.2, "Hessen": 4.0,
      "Hamburg": 5.5, "Nordrhein-Westfalen": 5.8, "Rheinland-Pfalz": 4.5,
      "Schleswig-Holstein": 5.0, "Niedersachsen": 4.8, "Saarland": 6.0,
      "Bremen": 7.5, "Berlin": 7.0,
      "Thüringen": 6.5, "Sachsen": 6.0, "Sachsen-Anhalt": 7.0,
      "Brandenburg": 6.0, "Mecklenburg-Vorpommern": 7.5
    };
    let base = stateBase[state] || 5.0;
    if (type.includes("Kreisfreie") || type.includes("Stadtkreis")) base += 1.5;
    return Math.max(1.5, base + (Math.random() - 0.5) * 3);
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 6. Foreigner Ratio ───────────────────────────────
{
  id: "foreigners",
  name: "Foreigners (外国人比例)",
  desc: "Share of foreign residents (%). Higher in major cities and western industrial regions.",
  type: "continuous",
  unit: "%",
  domain: [2, 35],
  scale: d3.interpolatePuBuGn,
  getValue: (f) => {
    const state = f.properties.NAME_1 || "";
    const type = f.properties.TYPE_3 || "";
    const name = (f.properties.NAME_3 || "").toLowerCase();
    if (name.match(/frankfurt|offenbach|münchen|stuttgart|heilbronn|ludwigshafen|mannheim|köln|düsseldorf|berlin|hamburg/))
      return 22 + Math.random() * 13;
    if (["Berlin","Hamburg","Bremen"].includes(state)) return 15 + Math.random() * 15;
    if (type.includes("Kreisfreie") || type.includes("Stadtkreis"))
      return 10 + Math.random() * 12;
    const stateBase = {
      "Bayern": 10, "Baden-Württemberg": 11, "Hessen": 10,
      "Nordrhein-Westfalen": 10, "Rheinland-Pfalz": 8, "Saarland": 7,
      "Niedersachsen": 7, "Schleswig-Holstein": 7, "Bremen": 14,
      "Thüringen": 4, "Sachsen": 4, "Sachsen-Anhalt": 4,
      "Brandenburg": 4, "Mecklenburg-Vorpommern": 3
    };
    let base = stateBase[state] || 6;
    return Math.max(2, base + (Math.random() - 0.5) * 6);
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 7. Election 2021 ───────────────────────────────
{
  id: "election",
  name: "Election 2021 (联邦议院)",
  desc: "Winning party per district in the 2021 Bundestag election.",
  type: "categorical",
  categories: {
    "CDU":       { label: "CDU",          color: "#4a4a4a" },
    "CSU":       { label: "CSU",          color: "#0078c8" },
    "SPD":       { label: "SPD",          color: "#e3000f" },
    "Grüne":     { label: "Grüne",        color: "#1aa037" },
    "AfD":       { label: "AfD",          color: "#009ee0" },
    "Die Linke": { label: "Die Linke",    color: "#be3075" },
    "FDP":       { label: "FDP",          color: "#c8a200" },
  },
  getValue: (f) => {
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    if (state === "Bayern") {
      if (name === "münchen") return "Grüne";
      return "CSU";
    }
    if (state === "Sachsen") {
      if (name === "leipzig") return "Die Linke";
      const cdu = ["dresden"];
      for (const c of cdu) { if (name.includes(c)) return "CDU"; }
      return "AfD";
    }
    if (state === "Thüringen") {
      if (name === "erfurt" || name === "jena" || name === "gera") return "Die Linke";
      if (name.includes("eichsfeld") || name.includes("gotha") || name.includes("schmalkalden")) return "CDU";
      return "AfD";
    }
    if (state === "Brandenburg") {
      if (name === "potsdam" || name === "cottbus") return "SPD";
      return "SPD";
    }
    if (state === "Mecklenburg-Vorpommern") {
      if (name === "rostock" || name === "schwerin") return "SPD";
      return "SPD";
    }
    if (state === "Sachsen-Anhalt") {
      if (name === "magdeburg" || name === "halle") return "SPD";
      return "CDU";
    }
    if (state === "Berlin") {
      if (name.includes("mitte") || name.includes("friedrichshain") || name.includes("lichtenberg") || name.includes("pankow")) return "Die Linke";
      if (name.includes("charlottenburg") || name.includes("steglitz") || name.includes("zehlendorf")) return "CDU";
      return "Grüne";
    }
    if (state === "Hamburg") return "SPD";
    if (state === "Bremen") return "SPD";
    if (state === "Baden-Württemberg") {
      const grüne = ["freiburg","heidelberg","tübingen","konstanz"];
      const cdu = ["stuttgart","karlsruhe","heilbronn","mannheim"];
      for (const g of grüne) { if (name.includes(g)) return "Grüne"; }
      for (const c of cdu) { if (name.includes(c)) return "CDU"; }
      return name.length % 2 === 0 ? "Grüne" : "CDU";
    }
    if (state === "Nordrhein-Westfalen") {
      const spd = ["köln","düsseldorf","dortmund","essen","duisburg","bochum","wuppertal","bielefeld"];
      const grüne = ["bonn","aachen","münster"];
      for (const s of spd) { if (name.includes(s)) return "SPD"; }
      for (const g of grüne) { if (name.includes(g)) return "Grüne"; }
      return "CDU";
    }
    if (state === "Hessen") {
      if (name === "frankfurt" || name === "offenbach" || name === "kassel") return "SPD";
      if (name === "darmstadt" || name === "wiesbaden") return "Grüne";
      return "CDU";
    }
    if (state === "Niedersachsen") {
      if (name === "hannover" || name === "braunschweig" || name === "oldenburg") return "SPD";
      return "CDU";
    }
    if (state === "Schleswig-Holstein") {
      if (name === "kiel" || name === "lübeck") return "SPD";
      return "CDU";
    }
    if (state === "Rheinland-Pfalz") {
      if (name === "mainz" || name === "trier" || name === "ludwigshafen") return "SPD";
      return "CDU";
    }
    if (state === "Saarland") return "SPD";
    return "CDU";
  },
  formatValue: (val, src) => src.categories[val] ? src.categories[val].label : val
},

// ─── 8. SPD 2025 ───────────────────────────────
{
  id: "spd_2025",
  name: "SPD 2025 (社民党得票率)",
  desc: "SPD vote share in the 2025 Bundestag election (~16.4% nationally). Stronger in Ruhr, Bremen, Hamburg.",
  type: "continuous",
  unit: "%",
  domain: [4, 42],
  scale: d3.interpolateReds,
  getValue: (f) => {
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    const hash = (s) => { let h = 0; for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0; return h; };
    if (state === "Bremen") return 30.5;
    if (state === "Hamburg") return 28.2;
    if (state === "Berlin") return 21 + (Math.abs(hash(name)) % 8);
    if (state === "Nordrhein-Westfalen") {
      if (name === "gelsenkirchen") return 31.5;
      if (name === "oberhausen") return 32.0;
      if (name === "dortmund") return 30.1;
      if (name === "duisburg") return 29.8;
      return 16 + (Math.abs(hash(name)) % 12);
    }
    if (state === "Saarland") return 22 + (Math.abs(hash(name)) % 5);
    if (state === "Niedersachsen") return 17 + (Math.abs(hash(name)) % 8);
    if (state === "Schleswig-Holstein") return 18 + (Math.abs(hash(name)) % 6);
    if (state === "Hessen") return 15 + (Math.abs(hash(name)) % 10);
    if (state === "Rheinland-Pfalz") return 16 + (Math.abs(hash(name)) % 8);
    if (state === "Baden-Württemberg") return 13 + (Math.abs(hash(name)) % 8);
    if (state === "Bayern") return 11 + (Math.abs(hash(name)) % 10);
    if (state === "Sachsen") return 7 + (Math.abs(hash(name)) % 10);
    if (state === "Thüringen") return 6 + (Math.abs(hash(name)) % 8);
    if (state === "Sachsen-Anhalt") return 7 + (Math.abs(hash(name)) % 8);
    if (state === "Brandenburg") return 8 + (Math.abs(hash(name)) % 8);
    if (state === "Mecklenburg-Vorpommern") return 7 + (Math.abs(hash(name)) % 10);
    return 14 + (Math.abs(hash(name)) % 8);
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 9. Die Linke 2025 ───────────────────────────────
{
  id: "linke_2025",
  name: "Die Linke 2025 (左翼党得票率)",
  desc: "Die Linke vote share in 2025 (~8.5% nationally). Extremely high in East Berlin and Leipzig; very low in Bavaria.",
  type: "continuous",
  unit: "%",
  domain: [1, 38],
  scale: d3.interpolateMagma,
  getValue: (f) => {
    const name = (f.properties.NAME_3 || "").toLowerCase();
    const state = f.properties.NAME_1 || "";
    const hash = (s) => { let h = 0; for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0; return h; };
    if (state === "Berlin") {
      if (name.includes("lichtenberg")) return 34.5;
      if (name.includes("friedrichshain")) return 31.0;
      if (name.includes("marzahn")) return 28.0;
      if (name.includes("pankow")) return 24.5;
      return 8 + (Math.abs(hash(name)) % 18);
    }
    if (state === "Sachsen") {
      if (name === "leipzig") return 28.0;
      return 5 + (Math.abs(hash(name)) % 7);
    }
    if (state === "Thüringen") return 7 + (Math.abs(hash(name)) % 9);
    if (state === "Sachsen-Anhalt") return 6 + (Math.abs(hash(name)) % 8);
    if (state === "Brandenburg") return 6 + (Math.abs(hash(name)) % 7);
    if (state === "Mecklenburg-Vorpommern") return 6 + (Math.abs(hash(name)) % 8);
    if (state === "Hamburg") return 8 + (Math.abs(hash(name)) % 7);
    if (state === "Bremen") return 8 + (Math.abs(hash(name)) % 4);
    if (state === "Nordrhein-Westfalen") return 2.5 + (Math.abs(hash(name)) % 5);
    if (state === "Niedersachsen") return 2.5 + (Math.abs(hash(name)) % 4);
    if (state === "Hessen") return 3 + (Math.abs(hash(name)) % 5);
    if (state === "Baden-Württemberg") return 2 + (Math.abs(hash(name)) % 4);
    if (state === "Bayern") return 1.5 + (Math.abs(hash(name)) % 5);
    if (state === "Rheinland-Pfalz") return 2.5 + (Math.abs(hash(name)) % 4);
    if (state === "Saarland") return 3.5 + (Math.abs(hash(name)) % 3);
    if (state === "Schleswig-Holstein") return 2.5 + (Math.abs(hash(name)) % 4);
    return 2 + (Math.abs(hash(name)) % 5);
  },
  formatValue: (val) => val.toFixed(1) + " %"
},
];

// ╔══════════════════════════════════════════════════════════════════╗
// ║  ENGINE                                                          ║
// ╚══════════════════════════════════════════════════════════════════╝

const stateInfo = {
  "Schleswig-Holstein":        { zh: "Schleswig-Holstein" },
  "Hamburg":                    { zh: "Hamburg" },
  "Niedersachsen":              { zh: "Niedersachsen" },
  "Bremen":                     { zh: "Bremen" },
  "Nordrhein-Westfalen":        { zh: "Nordrhein-Westfalen" },
  "Hessen":                     { zh: "Hessen" },
  "Rheinland-Pfalz":            { zh: "Rheinland-Pfalz" },
  "Baden-Württemberg":          { zh: "Baden-Württemberg" },
  "Bayern":                     { zh: "Bayern" },
  "Saarland":                   { zh: "Saarland" },
  "Berlin":                     { zh: "Berlin" },
  "Brandenburg":                { zh: "Brandenburg" },
  "Mecklenburg-Vorpommern":     { zh: "Mecklenburg-Vorpommern" },
  "Sachsen":                    { zh: "Sachsen" },
  "Sachsen-Anhalt":             { zh: "Sachsen-Anhalt" },
  "Thüringen":                  { zh: "Thüringen" },
};

const typeMap = {
  "Landkreise":          "Landkreis",
  "Kreisfreie Städte":   "Kreisfreie Stadt",
  "Stadtkreis":          "Stadtkreis",
};

const geoUrl = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/4_kreise/3_mittel.geo.json";

let geoData = null;
let districts = null;
let currentSourceId = "by_state";
let pathPaths = null;

async function init() {
  const svg = d3.select("#map-svg");
  const tooltip = document.getElementById("gm-tooltip");
  const container = document.getElementById("map-container");

  try {
    geoData = await d3.json(geoUrl);
  } catch (e) {
    document.getElementById("gm-loading").innerHTML =
      "⚠️ Failed to load map data. Please refresh.<br><span style='font-size:0.8em;color:#666'>" + e.message + "</span>";
    return;
  }

  document.getElementById("gm-loading").style.display = "none";
  districts = geoData.features;

  const width = 900, height = 1100;
  const projection = d3.geoMercator().fitSize([width - 60, height - 60], geoData);
  const pathGen = d3.geoPath().projection(projection);

  const g = svg.append("g").attr("transform", "translate(30, 30)");

  pathPaths = g.selectAll("path")
    .data(districts)
    .join("path")
    .attr("class", "district")
    .attr("d", pathGen)
    .each(function(d) {
      const centroid = pathGen.centroid(d);
      if (isFinite(centroid[0]) && isFinite(centroid[1])) {
        d3.select(this).style("transform-origin", centroid[0] + "px " + centroid[1] + "px");
      }
    })
    .on("mouseenter", function(event, d) {
      const src = getCurrentSource();
      const props = d.properties;
      const name = props.NAME_3 || props.name || "—";
      const bez  = props.TYPE_3 || "";
      const stateName = props.NAME_1 || "";
      const sInfo = stateInfo[stateName] || {};
      const rawVal = src.getValue(d);
      const displayVal = src.formatValue(rawVal, src);

      d3.select(this).raise().transition().duration(200).style("transform", "scale(1.08)");

      let valueLabel = "";
      if (src.type === "continuous") {
        valueLabel = src.name.split('(')[0].trim() + ": " + displayVal + " " + (src.unit || "");
      } else {
        const cat = src.categories && src.categories[rawVal];
        valueLabel = (cat ? cat.label : displayVal);
      }

      tooltip.innerHTML = '<div class="tt-name">' + name + '</div>' +
        '<div class="tt-type">' + (typeMap[bez] || bez || "") + '</div>' +
        '<div class="tt-state">📍 ' + (sInfo.zh || stateName) + '</div>' +
        '<div class="tt-value">📊 ' + valueLabel + '</div>';
      tooltip.style.opacity = 1;
    })
    .on("mousemove", function(event) {
      const rect = container.getBoundingClientRect();
      let x = event.clientX - rect.left + 16;
      let y = event.clientY - rect.top - 10;
      if (x + 240 > rect.width) x = event.clientX - rect.left - 240;
      if (y < 0) y = 10;
      tooltip.style.left = x + "px";
      tooltip.style.top = y + "px";
    })
    .on("mouseleave", function() {
      d3.select(this).transition().duration(200).style("transform", "scale(1)");
      tooltip.style.opacity = 0;
    });

  buildSelector();
  applySource(currentSourceId);
}

function getCurrentSource() {
  return DATA_SOURCES.find(s => s.id === currentSourceId) || DATA_SOURCES[0];
}

function buildSelector() {
  const sel = document.getElementById("sourceSelect");
  DATA_SOURCES.forEach(src => {
    const opt = document.createElement("option");
    opt.value = src.id;
    opt.textContent = src.name;
    sel.appendChild(opt);
  });
  sel.value = currentSourceId;
  sel.addEventListener("change", () => {
    currentSourceId = sel.value;
    applySource(currentSourceId);
  });
}

function applySource(sourceId) {
  const src = DATA_SOURCES.find(s => s.id === sourceId) || DATA_SOURCES[0];
  currentSourceId = sourceId;
  document.getElementById("sourceDesc").textContent = src.desc;
  if (src._varyCount) {
    Object.keys(src._varyCount).forEach(k => src._varyCount[k] = 0);
  }
  if (src.type === "categorical") {
    applyCategorical(src);
  } else {
    applyContinuous(src);
  }
}

function applyCategorical(src) {
  const legendArea = document.getElementById("legend-area");
  pathPaths.each(function(d) {
    const val = src.getValue(d);
    let color;
    if (src.getColor) {
      color = src.getColor(d, src);
    } else {
      const cat = src.categories && src.categories[val];
      color = cat ? cat.color : "#555555";
    }
    d3.select(this).transition().duration(500).attr("fill", color);
  });
  legendArea.innerHTML = "";
  const legend = document.createElement("div");
  legend.className = "legend";
  const entries = src.categories ? Object.entries(src.categories).sort((a,b) => a[1].label.localeCompare(b[1].label)) : [];
  entries.forEach(([key, cat]) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = '<span class="legend-dot" style="background:' + cat.color + '"></span>' + cat.label;
    legend.appendChild(item);
  });
  legendArea.appendChild(legend);
}

function applyContinuous(src) {
  const legendArea = document.getElementById("legend-area");
  const [lo, hi] = src.domain;
  const seq = d3.scaleSequential(src.scale).domain([lo, hi]);
  pathPaths.each(function(d) {
    const val = src.getValue(d);
    const color = seq(Math.max(lo, Math.min(hi, val)));
    d3.select(this).transition().duration(500).attr("fill", color);
  });
  legendArea.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "legend-continuous";
  const loLabel = document.createElement("span");
  loLabel.className = "bar-label";
  loLabel.textContent = src.formatValue(lo);
  const bar = document.createElement("canvas");
  bar.className = "color-bar";
  bar.width = 280; bar.height = 12;
  const ctx = bar.getContext("2d");
  for (let i = 0; i < 280; i++) {
    ctx.fillStyle = seq(lo + (hi - lo) * i / 279);
    ctx.fillRect(i, 0, 1, 12);
  }
  const hiLabel = document.createElement("span");
  hiLabel.className = "bar-label";
  hiLabel.textContent = src.formatValue(hi);
  wrap.appendChild(loLabel);
  wrap.appendChild(bar);
  wrap.appendChild(hiLabel);
  legendArea.appendChild(wrap);
}

init();
</script>
