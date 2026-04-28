---
layout: page
full-width: true
---

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">ls -la ~/vibing/</span>
      <span style="color: var(--oc-text-tertiary);">// coding toys</span>
    </div>
    <div class="oc-topbar-right">
      <span class="oc-status-text" style="color: var(--oc-text-tertiary);">4 items</span>
    </div>
  </div>

  <!-- Description -->
  <div class="oc-vibing-desc oc-fade-in">
    A collection of small projects I built through <span class="oc-vibing-highlight">Vibe Coding</span> — keeping a record makes it even more rewarding ✨
  </div>

  <!-- Gallery Grid -->
  <div style="padding: 10px; flex: 1;">
    <div class="oc-gallery">

      <!-- Toy 1 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-1">
        <div class="oc-gallery-img-wrap">
          <div class="oc-spinner"></div>
          <img src="/assets/img/vibing-2048.jpg" alt="2048" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">2048</div>
          <div class="oc-gallery-caption">Classic 2048 sliding puzzle game</div>
          <div class="oc-gallery-meta"><a href="/2048" style="color:var(--oc-blue);text-decoration:none;">Play →</a></div>
        </div>
      </div>

      <!-- Toy 2 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-2">
        <div class="oc-gallery-img-wrap">
          <div class="oc-spinner"></div>
          <img src="/assets/img/vibing-flappybird.jpg" alt="Flappy Bird" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">flappy_bird</div>
          <div class="oc-gallery-caption">Flappy Bird clone built with Canvas API</div>
          <div class="oc-gallery-meta"><a href="/flappybird" style="color:var(--oc-blue);text-decoration:none;">Play →</a></div>
        </div>
      </div>

      <!-- Toy 3 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-3">
        <div class="oc-gallery-img-wrap">
          <div class="oc-spinner"></div>
          <img src="/assets/img/vibing-germanymap.jpg" alt="Germany Map" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">germany_map</div>
          <div class="oc-gallery-caption">Interactive choropleth map of 401 German districts</div>
          <div class="oc-gallery-meta"><a href="/germany-map" style="color:var(--oc-blue);text-decoration:none;">View →</a></div>
        </div>
      </div>

      <!-- Toy 4 -->
      <div class="oc-gallery-item oc-fade-in oc-delay-4">
        <div class="oc-gallery-img-wrap">
          <div class="oc-spinner"></div>
          <img src="/assets/img/vibing-japanmap.jpg" alt="Japan Map" class="oc-gallery-img" loading="lazy">
        </div>
        <div class="oc-gallery-info">
          <div class="oc-gallery-filename">japan_map</div>
          <div class="oc-gallery-caption">Interactive choropleth map of 47 Japanese prefectures</div>
          <div class="oc-gallery-meta"><a href="/japan-map" style="color:var(--oc-blue);text-decoration:none;">View →</a></div>
        </div>
      </div>

    </div>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-5">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">4</span>
        <span class="oc-stat-label">toys</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">0</span>
        <span class="oc-stat-label">done</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/vibing/ <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
/* ============================================
   Vibing Gallery - OpenCode Geek Style
   ============================================ */

.oc-vibing-desc {
  padding: 16px 20px;
  font-size: 15px;
  color: var(--oc-text);
  line-height: 1.7;
  border-bottom: 1px solid var(--oc-border);
}

.oc-vibing-highlight {
  color: var(--oc-blue);
  font-weight: 600;
}
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
  transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), filter 0.3s ease, opacity 0.4s ease;
  filter: brightness(0.92) contrast(1.02);
  opacity: 0;
}

.oc-gallery-img.oc-loaded {
  opacity: 1;
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

/* Loading spinner */
.oc-spinner {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 28px;
  height: 28px;
  border: 2.5px solid rgba(255, 255, 255, 0.15);
  border-top-color: rgba(255, 255, 255, 0.7);
  border-radius: 50%;
  animation: oc-spin 0.8s linear infinite;
  z-index: 2;
  transition: opacity 0.3s ease;
}

.oc-gallery-img-wrap.oc-img-loaded .oc-spinner {
  opacity: 0;
  pointer-events: none;
}

@keyframes oc-spin {
  to { transform: translate(-50%, -50%) rotate(360deg); }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  var imgs = document.querySelectorAll('.oc-gallery-img');
  imgs.forEach(function(img) {
    function onLoaded() {
      img.classList.add('oc-loaded');
      img.closest('.oc-gallery-img-wrap').classList.add('oc-img-loaded');
    }
    if (img.complete && img.naturalWidth > 0) {
      onLoaded();
    } else {
      img.addEventListener('load', onLoaded);
      img.addEventListener('error', onLoaded);
    }
  });
});
</script>
