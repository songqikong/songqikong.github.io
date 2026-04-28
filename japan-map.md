---
layout: page
full-width: true
title: "Japan Map"
---

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">./japan_map</span>
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
      <span class="gm-logo">&gt;_japan_map</span>
      <span class="gm-ver">v1.0</span>
      <span class="gm-subtitle">47 Prefectures of Japan — hover to inspect</span>
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
      <svg id="map-svg" viewBox="0 0 1000 900" preserveAspectRatio="xMidYMid meet"></svg>
      <div id="gm-tooltip"></div>
    </div>

    <div id="legend-area"></div>

    <div class="gm-info-bar">47 Prefectures (都道府県)</div>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-2">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">47</span>
        <span class="oc-stat-label">prefectures</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">8</span>
        <span class="oc-stat-label">regions</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/vibing/japan_map <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
/* ============================================
   Japan Map - OpenCode Geek Style
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
  min-width: 260px;
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
  width: 1000px;
  max-width: 95vw;
}

.gm-map-container svg {
  width: 100%;
  height: auto;
  filter: drop-shadow(0 0 20px rgba(37, 99, 235, 0.1));
}

.district {
  stroke: rgba(88, 166, 255, 0.25);
  stroke-width: 0.8;
  cursor: pointer;
  transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1),
              filter 0.2s ease,
              stroke-width 0.2s ease,
              fill 0.5s ease;
  transform-origin: center;
}

.district:hover {
  stroke: #58A6FF;
  stroke-width: 2;
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
  max-width: 1000px;
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
  .gm-source-select { min-width: 200px; font-size: 11px; }
  .gm-source-desc { font-size: 10px; }
}
</style>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ╔══════════════════════════════════════════════════════════════════╗
// ║  REGION / PREFECTURE MAPPING                                     ║
// ╚══════════════════════════════════════════════════════════════════╝

const PREFECTURE_INFO = {
  "北海道":   { en: "Hokkaido",    region: "北海道",   regionEn: "Hokkaido" },
  "青森県":   { en: "Aomori",      region: "東北",     regionEn: "Tohoku" },
  "岩手県":   { en: "Iwate",       region: "東北",     regionEn: "Tohoku" },
  "宮城県":   { en: "Miyagi",      region: "東北",     regionEn: "Tohoku" },
  "秋田県":   { en: "Akita",       region: "東北",     regionEn: "Tohoku" },
  "山形県":   { en: "Yamagata",    region: "東北",     regionEn: "Tohoku" },
  "福島県":   { en: "Fukushima",   region: "東北",     regionEn: "Tohoku" },
  "茨城県":   { en: "Ibaraki",     region: "関東",     regionEn: "Kanto" },
  "栃木県":   { en: "Tochigi",     region: "関東",     regionEn: "Kanto" },
  "群馬県":   { en: "Gunma",       region: "関東",     regionEn: "Kanto" },
  "埼玉県":   { en: "Saitama",     region: "関東",     regionEn: "Kanto" },
  "千葉県":   { en: "Chiba",       region: "関東",     regionEn: "Kanto" },
  "東京都":   { en: "Tokyo",       region: "関東",     regionEn: "Kanto" },
  "神奈川県": { en: "Kanagawa",    region: "関東",     regionEn: "Kanto" },
  "新潟県":   { en: "Niigata",     region: "中部",     regionEn: "Chubu" },
  "富山県":   { en: "Toyama",      region: "中部",     regionEn: "Chubu" },
  "石川県":   { en: "Ishikawa",    region: "中部",     regionEn: "Chubu" },
  "福井県":   { en: "Fukui",       region: "中部",     regionEn: "Chubu" },
  "山梨県":   { en: "Yamanashi",   region: "中部",     regionEn: "Chubu" },
  "長野県":   { en: "Nagano",      region: "中部",     regionEn: "Chubu" },
  "岐阜県":   { en: "Gifu",        region: "中部",     regionEn: "Chubu" },
  "静岡県":   { en: "Shizuoka",    region: "中部",     regionEn: "Chubu" },
  "愛知県":   { en: "Aichi",       region: "中部",     regionEn: "Chubu" },
  "三重県":   { en: "Mie",         region: "近畿",     regionEn: "Kansai" },
  "滋賀県":   { en: "Shiga",       region: "近畿",     regionEn: "Kansai" },
  "京都府":   { en: "Kyoto",       region: "近畿",     regionEn: "Kansai" },
  "大阪府":   { en: "Osaka",       region: "近畿",     regionEn: "Kansai" },
  "兵庫県":   { en: "Hyogo",       region: "近畿",     regionEn: "Kansai" },
  "奈良県":   { en: "Nara",        region: "近畿",     regionEn: "Kansai" },
  "和歌山県": { en: "Wakayama",    region: "近畿",     regionEn: "Kansai" },
  "鳥取県":   { en: "Tottori",     region: "中国",     regionEn: "Chugoku" },
  "島根県":   { en: "Shimane",     region: "中国",     regionEn: "Chugoku" },
  "岡山県":   { en: "Okayama",     region: "中国",     regionEn: "Chugoku" },
  "広島県":   { en: "Hiroshima",   region: "中国",     regionEn: "Chugoku" },
  "山口県":   { en: "Yamaguchi",   region: "中国",     regionEn: "Chugoku" },
  "徳島県":   { en: "Tokushima",   region: "四国",     regionEn: "Shikoku" },
  "香川県":   { en: "Kagawa",      region: "四国",     regionEn: "Shikoku" },
  "愛媛県":   { en: "Ehime",       region: "四国",     regionEn: "Shikoku" },
  "高知県":   { en: "Kochi",       region: "四国",     regionEn: "Shikoku" },
  "福岡県":   { en: "Fukuoka",     region: "九州",     regionEn: "Kyushu" },
  "佐賀県":   { en: "Saga",        region: "九州",     regionEn: "Kyushu" },
  "長崎県":   { en: "Nagasaki",    region: "九州",     regionEn: "Kyushu" },
  "熊本県":   { en: "Kumamoto",    region: "九州",     regionEn: "Kyushu" },
  "大分県":   { en: "Oita",        region: "九州",     regionEn: "Kyushu" },
  "宮崎県":   { en: "Miyazaki",    region: "九州",     regionEn: "Kyushu" },
  "鹿児島県": { en: "Kagoshima",   region: "九州",     regionEn: "Kyushu" },
  "沖縄県":   { en: "Okinawa",     region: "九州",     regionEn: "Kyushu" },
};

function getRegion(pref) {
  const info = PREFECTURE_INFO[pref];
  return info ? info.region : "Unknown";
}

function getEn(pref) {
  const info = PREFECTURE_INFO[pref];
  return info ? info.en : pref;
}

// Extra data not in GeoJSON
const PREF_DATA = {
  "北海道":   { pop: 5224000, area: 83424, density: 63 },
  "青森県":   { pop: 1236000, area: 9607, density: 129 },
  "岩手県":   { pop: 1216000, area: 15275, density: 80 },
  "宮城県":   { pop: 2305000, area: 7282, density: 317 },
  "秋田県":   { pop: 945000, area: 11638, density: 81 },
  "山形県":   { pop: 1068000, area: 9323, density: 115 },
  "福島県":   { pop: 1834000, area: 13784, density: 133 },
  "茨城県":   { pop: 2867000, area: 6097, density: 470 },
  "栃木県":   { pop: 1934000, area: 6408, density: 302 },
  "群馬県":   { pop: 1908000, area: 6362, density: 300 },
  "埼玉県":   { pop: 7345000, area: 3798, density: 1934 },
  "千葉県":   { pop: 6284000, area: 5158, density: 1218 },
  "東京都":   { pop: 14050000, area: 2194, density: 6403 },
  "神奈川県": { pop: 9237000, area: 2416, density: 3825 },
  "新潟県":   { pop: 2149000, area: 12584, density: 171 },
  "富山県":   { pop: 1036000, area: 4248, density: 244 },
  "石川県":   { pop: 1123000, area: 4186, density: 268 },
  "福井県":   { pop: 756000, area: 4190, density: 180 },
  "山梨県":   { pop: 809000, area: 4465, density: 181 },
  "長野県":   { pop: 2048000, area: 13562, density: 151 },
  "岐阜県":   { pop: 1897000, area: 10621, density: 179 },
  "静岡県":   { pop: 3573000, area: 7777, density: 459 },
  "愛知県":   { pop: 7542000, area: 5173, density: 1458 },
  "三重県":   { pop: 1687000, area: 5774, density: 292 },
  "滋賀県":   { pop: 1413000, area: 4017, density: 352 },
  "京都府":   { pop: 2513000, area: 4612, density: 545 },
  "大阪府":   { pop: 8838000, area: 1905, density: 4640 },
  "兵庫県":   { pop: 5465000, area: 8401, density: 651 },
  "奈良県":   { pop: 1318000, area: 3691, density: 357 },
  "和歌山県": { pop: 878000, area: 4725, density: 186 },
  "鳥取県":   { pop: 543000, area: 3507, density: 155 },
  "島根県":   { pop: 658000, area: 6708, density: 98 },
  "岡山県":   { pop: 1876000, area: 7114, density: 264 },
  "広島県":   { pop: 2744000, area: 8480, density: 324 },
  "山口県":   { pop: 1287000, area: 6112, density: 211 },
  "徳島県":   { pop: 700000, area: 4147, density: 169 },
  "香川県":   { pop: 942000, area: 1877, density: 502 },
  "愛媛県":   { pop: 1287000, area: 5676, density: 227 },
  "高知県":   { pop: 650000, area: 7104, density: 92 },
  "福岡県":   { pop: 5135000, area: 4987, density: 1030 },
  "佐賀県":   { pop: 806000, area: 2441, density: 330 },
  "長崎県":   { pop: 1240000, area: 4131, density: 300 },
  "熊本県":   { pop: 1687000, area: 7409, density: 228 },
  "大分県":   { pop: 1103000, area: 6341, density: 174 },
  "宮崎県":   { pop: 1012000, area: 7735, density: 131 },
  "鹿児島県": { pop: 1548000, area: 9187, density: 168 },
  "沖縄県":   { pop: 1467000, area: 2281, density: 643 },
};

// ╔══════════════════════════════════════════════════════════════════╗
// ║  DATA SOURCES                                                    ║
// ╚══════════════════════════════════════════════════════════════════╝

const DATA_SOURCES = [

// ─── 1. By Region (地方) ─────────────────────────────
{
  id: "by_region",
  name: "Region (地方)",
  desc: "Color by traditional region (地方). Japan has 8 regions spanning from Hokkaido to Kyushu+Okinawa.",
  type: "categorical",
  categories: {
    "北海道": { label: "Hokkaido 北海道",   color: "#4e79a7" },
    "東北":   { label: "Tohoku 東北",       color: "#f28e2b" },
    "関東":   { label: "Kanto 関東",        color: "#e15759" },
    "中部":   { label: "Chubu 中部",        color: "#76b7b2" },
    "近畿":   { label: "Kansai 近畿",       color: "#59a14f" },
    "中国":   { label: "Chugoku 中国",      color: "#edc948" },
    "四国":   { label: "Shikoku 四国",      color: "#b07aa1" },
    "九州":   { label: "Kyushu 九州",       color: "#ff9da7" },
  },
  getValue: (f) => getRegion(f.properties.nam_ja),
  formatValue: (val, src) => {
    const cat = src.categories[val];
    return cat ? cat.label : val;
  },
  _varyCount: {},
  getColor: function(f, src) {
    const key = getRegion(f.properties.nam_ja);
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

// ─── 2. Population (人口) ───────────────────────────────
{
  id: "population",
  name: "Population (人口)",
  desc: "Total population per prefecture. Tokyo and Kanagawa top the list; rural prefectures like Tottori and Shimane are the smallest.",
  type: "continuous",
  unit: "人",
  domain: [500000, 14000000],
  scale: d3.interpolateBlues,
  getValue: (f) => (PREF_DATA[f.properties.nam_ja] || {}).pop || 1000000,
  formatValue: (val) => (val / 10000).toFixed(0) + " 万人"
},

// ─── 3. Population Density (人口密度) ───────────────────────
{
  id: "density",
  name: "Pop. Density (人口密度)",
  desc: "Residents per km². Tokyo is off the charts at ~6,000/km²; Hokkaido is the sparsest at ~60/km².",
  type: "continuous",
  unit: "/km²",
  domain: [50, 6500],
  scale: d3.interpolateOrRd,
  getValue: (f) => (PREF_DATA[f.properties.nam_ja] || {}).density || 500,
  formatValue: (val) => Math.round(val) + " /km²"
},

// ─── 4. Area (面積) ───────────────────────────────
{
  id: "area",
  name: "Area (面積)",
  desc: "Land area in km². Hokkaido is by far the largest; Kagawa and Osaka are the smallest.",
  type: "continuous",
  unit: "km²",
  domain: [180, 8400],
  scale: d3.interpolateYlGn,
  getValue: (f) => (PREF_DATA[f.properties.nam_ja] || {}).area || 3000,
  formatValue: (val) => Math.round(val) + " km²"
},

// ─── 5. Aging Rate (高齢化率) ───────────────────────────────
{
  id: "aging",
  name: "Aging Rate (高齢化率)",
  desc: "Percentage of population aged 65+. Akita and Kochi exceed 35%; Okinawa and Saitama are below 27%.",
  type: "continuous",
  unit: "%",
  domain: [24, 39],
  scale: d3.interpolatePurples,
  getValue: (f) => {
    const pref = f.properties.nam_ja;
    const data = {
      "秋田県":37.8,"高知県":36.2,"島根県":35.9,"山形県":35.7,"岩手県":35.4,
      "鳥取県":34.8,"長崎県":34.5,"徳島県":34.2,"鹿児島県":34.0,"大分県":33.5,
      "山口県":33.2,"宮崎県":33.0,"和歌山県":32.8,"愛媛県":32.5,"福島県":32.2,
      "香川県":32.0,"熊本県":31.8,"石川県":31.5,"富山県":31.2,"広島県":31.0,
      "佐賀県":30.8,"岡山県":30.5,"兵庫県":30.2,"奈良県":30.0,"栃木県":29.8,
      "新潟県":29.6,"宮城県":29.4,"福井県":29.2,"青森県":29.0,"岐阜県":28.8,
      "三重県":28.6,"山梨県":28.4,"静岡県":28.2,"京都府":28.0,"千葉県":27.8,
      "北海道":27.6,"群馬県":27.4,"福岡県":27.2,"埼玉県":27.0,"大阪府":26.8,
      "長野県":26.6,"滋賀県":26.4,"東京都":26.2,"愛知県":26.0,"神奈川県":25.8,
      "茨城県":25.6,"沖縄県":24.5
    };
    return data[pref] || 28;
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 6. GDP per Capita (一人当たりGDP) ───────────────────────
{
  id: "gdp",
  name: "GDP per Capita (一人当たり県民経済計算)",
  desc: "Per-capita gross prefectural product (万円). Tokyo and Aichi (home of Toyota) lead; Okinawa is the lowest.",
  type: "continuous",
  unit: "万円",
  domain: [220, 780],
  scale: d3.interpolateYlGnBu,
  getValue: (f) => {
    const pref = f.properties.nam_ja;
    const data = {
      "東京都":750,"愛知県":560,"大阪府":480,"神奈川県":470,"千葉県":400,
      "兵庫県":390,"福岡県":380,"静岡県":420,"茨城県":370,"広島県":380,
      "埼玉県":380,"京都府":400,"群馬県":360,"栃木県":360,"三重県":370,
      "長野県":340,"山梨県":330,"滋賀県":360,"石川県":350,"富山県":350,
      "岡山県":350,"宮城県":360,"北海道":320,"新潟県":310,"福井県":370,
      "岐阜県":330,"熊本県":290,"鹿児島県":260,"奈良県":310,"和歌山県":300,
      "青森県":270,"山形県":290,"秋田県":280,"岩手県":270,"福島県":290,
      "鳥取県":280,"島根県":290,"山口県":310,"香川県":320,"徳島県":290,
      "愛媛県":290,"高知県":270,"佐賀県":280,"長崎県":270,"大分県":290,
      "宮崎県":260,"沖縄県":230
    };
    return data[pref] || 350;
  },
  formatValue: (val) => val.toFixed(0) + " 万円"
},

// ─── 7. LDP 2024 Election (自民党得票率) ───────────────────────
{
  id: "ldp_2024",
  name: "LDP 2024 (自民党得票率)",
  desc: "Liberal Democratic Party vote share in the 2024 House of Representatives election (~34% nationally). Stronger in rural areas; weaker in urban centers.",
  type: "continuous",
  unit: "%",
  domain: [18, 55],
  scale: d3.interpolateGreens,
  getValue: (f) => {
    const pref = f.properties.nam_ja;
    const data = {
      "秋田県":42,"山形県":44,"島根県":48,"鳥取県":43,"福井県":46,
      "富山県":45,"石川県":43,"山梨県":40,"長野県":38,"岐阜県":41,
      "静岡県":39,"三重県":42,"滋賀県":37,"奈良県":38,"和歌山県":41,
      "岡山県":43,"広島県":40,"山口県":44,"徳島県":42,"香川県":43,
      "愛媛県":42,"高知県":39,"佐賀県":41,"長崎県":40,"熊本県":39,
      "大分県":41,"宮崎県":40,"鹿児島県":41,"沖縄県":20,
      "青森県":38,"岩手県":35,"宮城県":37,"福島県":40,"茨城県":40,
      "栃木県":39,"群馬県":41,"埼玉県":33,"千葉県":32,"東京都":29,
      "神奈川県":28,"新潟県":38,"京都府":33,"大阪府":25,"兵庫県":32,
      "北海道":34,"愛知県":32,"福岡県":35
    };
    return data[pref] || 34;
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 8. CDP 2024 Election (立憲民主党得票率) ───────────────────────
{
  id: "cdp_2024",
  name: "CDP 2024 (立憲民主党得票率)",
  desc: "Constitutional Democratic Party vote share in 2024 (~22% nationally). Stronger in urban areas like Tokyo and Osaka.",
  type: "continuous",
  unit: "%",
  domain: [5, 38],
  scale: d3.interpolateRdYlBu,
  getValue: (f) => {
    const pref = f.properties.nam_ja;
    const data = {
      "北海道":26,"青森県":18,"岩手県":20,"宮城県":22,"秋田県":16,
      "山形県":15,"福島県":17,"茨城県":20,"栃木県":21,"群馬県":19,
      "埼玉県":27,"千葉県":26,"東京都":35,"神奈川県":33,"新潟県":19,
      "富山県":16,"石川県":18,"福井県":14,"山梨県":18,"長野県":22,
      "岐阜県":18,"静岡県":21,"愛知県":26,"三重県":18,"滋賀県":23,
      "京都府":28,"大阪府":32,"兵庫県":27,"奈良県":23,"和歌山県":17,
      "鳥取県":15,"島根県":14,"岡山県":18,"広島県":21,"山口県":17,
      "徳島県":16,"香川県":17,"愛媛県":17,"高知県":19,"福岡県":24,
      "佐賀県":17,"長崎県":18,"熊本県":20,"大分県":18,"宮崎県":17,
      "鹿児島県":18,"沖縄県":12
    };
    return data[pref] || 22;
  },
  formatValue: (val) => val.toFixed(1) + " %"
},

// ─── 9. Population Change 2015-2020 (人口増減率) ───────────────────────
{
  id: "pop_change",
  name: "Pop. Change 2015-20 (人口増減率)",
  desc: "Population change rate 2015→2020 (%). Only Tokyo, Okinawa, and a few others grew; most rural prefectures declined.",
  type: "continuous",
  unit: "%",
  domain: [-5.5, 4.0],
  scale: d3.interpolateRdYlGn,
  getValue: (f) => {
    const pref = f.properties.nam_ja;
    const data = {
      "東京都":3.8,"神奈川県":1.2,"埼玉県":1.0,"千葉県":0.5,"愛知県":1.5,
      "大阪府":0.2,"福岡県":0.8,"沖縄県":2.1,"滋賀県":0.3,"宮城県":0.1,
      "広島県":-0.3,"兵庫県":-0.8,"京都府":-0.5,"静岡県":-0.9,"茨城県":-1.0,
      "群馬県":-1.2,"栃木県":-0.8,"三重県":-1.0,"長野県":-0.7,"石川県":-0.4,
      "富山県":-1.1,"岐阜県":-1.3,"山梨県":-1.0,"福井県":-0.8,"奈良県":-1.5,
      "和歌山県":-2.0,"新潟県":-2.1,"北海道":-2.2,"青森県":-2.8,"岩手県":-2.5,
      "秋田県":-3.5,"山形県":-2.3,"福島県":-2.0,"鳥取県":-1.8,"島根県":-2.0,
      "岡山県":-0.9,"山口県":-2.0,"徳島県":-1.8,"香川県":-1.2,"愛媛県":-2.0,
      "高知県":-2.5,"佐賀県":-1.5,"長崎県":-2.3,"熊本県":-0.8,"大分県":-1.5,
      "宮崎県":-1.3,"鹿児島県":-1.8
    };
    return data[pref] || -1;
  },
  formatValue: (val) => (val > 0 ? "+" : "") + val.toFixed(1) + " %"
},

];

// ╔══════════════════════════════════════════════════════════════════╗
// ║  ENGINE                                                          ║
// ╚══════════════════════════════════════════════════════════════════╝

const geoUrl = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson";

let geoData = null;
let districts = null;
let currentSourceId = "by_region";
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

  const width = 1000, height = 900;
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
      const pref = props.nam_ja || "—";
      const enName = props.nam || "";
      const info = PREFECTURE_INFO[pref] || {};
      const pData = PREF_DATA[pref] || {};
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

      tooltip.innerHTML = '<div class="tt-name">' + pref + ' / ' + enName + '</div>' +
        '<div class="tt-type">📍 ' + (info.region || "") + ' (' + (info.regionEn || "") + ')</div>' +
        '<div class="tt-state">👥 ' + (pData.pop ? (pData.pop / 10000).toFixed(0) + '万人' : '—') + '</div>' +
        '<div class="tt-value">📊 ' + valueLabel + '</div>';
      tooltip.style.opacity = 1;
    })
    .on("mousemove", function(event) {
      const rect = container.getBoundingClientRect();
      let x = event.clientX - rect.left + 16;
      let y = event.clientY - rect.top - 10;
      if (x + 260 > rect.width) x = event.clientX - rect.left - 260;
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
