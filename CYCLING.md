---
layout: page
full-width: true
---

<link rel="stylesheet" href="{{ '/assets/css/apple-style.css' | relative_url }}">

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">ls -la ~/cycling/</span>
      <span style="color: var(--oc-text-tertiary);">// photo gallery</span>
    </div>
    <div class="oc-topbar-right">
      <span class="oc-status-text" style="color: var(--oc-text-tertiary);">4 items</span>
    </div>
  </div>

  <!-- Gallery Grid -->
  <div style="padding: 10px; flex: 1;">
    <div class="oc-gallery">

      <!-- Photo 1 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-1">
        <div class="oc-gallery-img-wrap">
          <img src="/assets/img/2024_6_5.jpg" alt="West Coast of Qingdao" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">2024_6_5.jpg</div>
          <div class="oc-gallery-caption">The seaside on the west coast of Qingdao</div>
          <div class="oc-gallery-meta">2024/06/05</div>
        </div>
      </div>

      <!-- Photo 2 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-2">
        <div class="oc-gallery-img-wrap">
          <img src="/assets/img/2024_4_24.jpg" alt="Mount Dazhu" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">2024_4_24.jpg</div>
          <div class="oc-gallery-caption">At the foot of Mount Dazhu</div>
          <div class="oc-gallery-meta">2024/04/24</div>
        </div>
      </div>

      <!-- Photo 3 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-3">
        <div class="oc-gallery-img-wrap">
          <img src="/assets/img/bike-2.jpg" alt="Spring Bridge" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">bike-2.jpg</div>
          <div class="oc-gallery-caption">Hexi District, Tianjin — Spring Bridge over Haihe River</div>
          <div class="oc-gallery-meta">2023/04/24</div>
        </div>
      </div>

      <!-- Photo 4 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-4">
        <div class="oc-gallery-img-wrap">
          <img src="/assets/img/bike-1.png" alt="My Bicycle" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">bike-1.png</div>
          <div class="oc-gallery-caption">Heping District, Tianjin — My Bicycle</div>
          <div class="oc-gallery-meta">2023/04/09</div>
        </div>
      </div>

    </div>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-5">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">4</span>
        <span class="oc-stat-label">photos</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">2</span>
        <span class="oc-stat-label">cities</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/cycling/ <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
/* ============================================
   Cycling Gallery - OpenCode Geek Style
   ============================================ */
.oc-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
}

.oc-gallery-item {
  background: var(--oc-bg-alt);
  border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius);
  overflow: hidden;
  transition: all var(--oc-transition);
}

.oc-gallery-item:hover {
  border-color: var(--oc-blue);
  box-shadow: 0 0 0 1px var(--oc-blue-border), 0 4px 16px var(--oc-blue-glow);
  transform: translateY(-2px);
}

.oc-gallery-img-wrap {
  width: 100%;
  height: 220px;
  overflow: hidden;
  position: relative;
  background: #0F172A;
}

.oc-gallery-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), filter 0.3s ease;
  filter: brightness(0.92) contrast(1.02);
}

.oc-gallery-item:hover .oc-gallery-img {
  transform: scale(1.05);
  filter: brightness(1) contrast(1);
}

/* Terminal-style file info overlay */
.oc-gallery-img-wrap::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 32px;
  background: linear-gradient(to bottom, rgba(15, 23, 42, 0.5), transparent);
  pointer-events: none;
}

.oc-gallery-info {
  padding: 10px 12px;
}

.oc-gallery-filename {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-blue);
  margin-bottom: 2px;
}

.oc-gallery-filename::before {
  content: '▸ ';
}

.oc-gallery-caption {
  font-size: 13px;
  color: var(--oc-text);
  font-weight: 500;
  line-height: 1.4;
}

.oc-gallery-meta {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  margin-top: 4px;
}

.oc-gallery-meta::before {
  content: '// ';
}

@media (max-width: 700px) {
  .oc-gallery {
    grid-template-columns: 1fr;
  }
  .oc-gallery-img-wrap {
    height: 200px;
  }
}
</style>
