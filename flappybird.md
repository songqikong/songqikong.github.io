---
layout: page
full-width: true
title: "Flappy Bird"
---

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">./flappy_bird</span>
      <span style="color: var(--oc-text-tertiary);">// arcade game</span>
    </div>
    <div class="oc-topbar-right">
      <div class="oc-status-dot"></div>
      <span class="oc-status-text">running</span>
    </div>
  </div>

  <div style="flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 16px;">
    <!-- Game Header -->
    <div class="fb-header oc-fade-in oc-delay-1">
      <div class="fb-title">
        <span class="fb-logo">&gt;_flappy</span>
        <span class="fb-ver">v1.0</span>
      </div>
      <div class="fb-scores">
        <div class="fb-score-box">
          <div class="fb-score-label">SCORE</div>
          <div class="fb-score-value" id="fb-score">0</div>
        </div>
        <div class="fb-score-box">
          <div class="fb-score-label">BEST</div>
          <div class="fb-score-value" id="fb-best">0</div>
        </div>
      </div>
    </div>

    <!-- Canvas -->
    <div class="fb-canvas-wrap oc-fade-in oc-delay-2">
      <canvas id="fb-canvas" width="320" height="480"></canvas>
      <div class="fb-overlay" id="fb-overlay">
        <div class="fb-overlay-icon">🐦</div>
        <div class="fb-overlay-msg" id="fb-overlay-msg">FLAPPY BIRD</div>
        <div class="fb-overlay-sub" id="fb-overlay-sub">Press SPACE or Tap to start</div>
        <button class="fb-btn" id="fb-start-btn">$ play</button>
      </div>
    </div>

    <!-- Controls -->
    <div class="fb-controls oc-fade-in oc-delay-3">
      <button class="fb-btn" onclick="flappyGame.restart()">$ restart</button>
      <div class="fb-hint">
        <span class="fb-key">SPACE</span>
        or tap to flap
      </div>
    </div>

    <a href="/VIBING" class="fb-back oc-fade-in oc-delay-4">← cd ~/vibing/</a>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-5">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">∞</span>
        <span class="oc-stat-label">pipes</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">1</span>
        <span class="oc-stat-label">bird</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/vibing/flappy_bird <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
/* ============================================
   Flappy Bird - OpenCode Geek Style
   ============================================ */

.fb-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  max-width: 340px;
  margin-bottom: 12px;
}

.fb-title {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.fb-logo {
  font-family: var(--oc-mono);
  font-size: 24px;
  font-weight: 700;
  color: var(--oc-blue);
  letter-spacing: -1px;
}

.fb-ver {
  font-family: var(--oc-mono);
  font-size: 10px;
  color: var(--oc-text-tertiary);
  padding: 1px 6px;
  border: 1px solid var(--oc-border);
  border-radius: 4px;
}

.fb-scores {
  display: flex;
  gap: 6px;
}

.fb-score-box {
  background: var(--oc-bg-terminal);
  border: 1px solid var(--oc-border);
  border-radius: 6px;
  padding: 4px 14px;
  text-align: center;
  min-width: 64px;
}

.fb-score-label {
  font-family: var(--oc-mono);
  font-size: 9px;
  font-weight: 600;
  color: var(--oc-text-tertiary);
  letter-spacing: 1.5px;
}

.fb-score-value {
  font-family: var(--oc-mono);
  font-size: 18px;
  font-weight: 700;
  color: var(--oc-blue-lighter);
}

/* Canvas wrap */
.fb-canvas-wrap {
  position: relative;
  width: 320px;
  height: 480px;
  border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius);
  overflow: hidden;
  background: var(--oc-bg-terminal);
}

.fb-canvas-wrap canvas {
  display: block;
  width: 320px;
  height: 480px;
}

/* Overlay */
.fb-overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(13, 17, 23, 0.88);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
  backdrop-filter: blur(4px);
}

.fb-overlay-icon {
  font-size: 40px;
  margin-bottom: 8px;
}

.fb-overlay-msg {
  font-family: var(--oc-mono);
  font-size: 20px;
  font-weight: 700;
  color: var(--oc-text);
  margin-bottom: 6px;
  letter-spacing: 2px;
}

.fb-overlay-sub {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  margin-bottom: 16px;
}

/* Buttons */
.fb-btn {
  font-family: var(--oc-mono);
  font-size: 12px;
  font-weight: 600;
  background: var(--oc-blue-bg);
  color: var(--oc-blue);
  border: 1px solid var(--oc-blue-border);
  border-radius: 6px;
  padding: 8px 20px;
  cursor: pointer;
  transition: all var(--oc-transition);
}

.fb-btn:hover {
  background: var(--oc-blue);
  color: #fff;
  border-color: var(--oc-blue);
  box-shadow: 0 0 12px var(--oc-blue-glow);
}

.fb-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 14px;
  width: 100%;
  max-width: 340px;
  justify-content: space-between;
}

.fb-hint {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  display: flex;
  align-items: center;
  gap: 4px;
}

.fb-key {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 22px;
  padding: 0 8px;
  border: 1px solid var(--oc-border);
  border-radius: 4px;
  font-size: 11px;
  color: var(--oc-text-secondary);
  background: var(--oc-bg-alt);
}

.fb-back {
  font-family: var(--oc-mono);
  font-size: 12px;
  color: var(--oc-text-tertiary);
  text-decoration: none;
  margin-top: 14px;
  transition: color var(--oc-transition);
}

.fb-back:hover {
  color: var(--oc-blue);
}
</style>

<script>
var flappyGame = (function() {
  var canvas, ctx;
  var W = 320, H = 480;

  // Game state
  var STATE_IDLE = 0, STATE_PLAYING = 1, STATE_DEAD = 2;
  var state = STATE_IDLE;
  var score = 0;
  var best = parseInt(localStorage.getItem('bestFlappy') || '0');
  var frameCount = 0;

  // Bird
  var bird = { x: 80, y: H / 2, vy: 0, radius: 14, angle: 0 };

  // Physics
  var GRAVITY = 0.3;
  var FLAP_POWER = -6;
  var TERMINAL_VEL = 8;

  // Pipes
  var pipes = [];
  var PIPE_WIDTH = 48;
  var PIPE_GAP = 130;
  var PIPE_SPEED = 2.2;
  var PIPE_INTERVAL = 95; // frames between pipes

  // Ground
  var GROUND_H = 60;
  var groundX = 0;

  // Colors (OpenCode terminal palette)
  var COL = {
    sky1: '#0F172A',
    sky2: '#1E293B',
    ground1: '#1C2128',
    ground2: '#30363D',
    groundLine: 'rgba(37,99,235,0.3)',
    pipe: '#2563EB',
    pipeBorder: 'rgba(88,166,255,0.5)',
    pipeHighlight: 'rgba(88,166,255,0.15)',
    pipeCap: 'rgba(37,99,235,0.6)',
    bird: '#58A6FF',
    birdBorder: '#A5D6FF',
    birdWing: 'rgba(88,166,255,0.5)',
    birdEye: '#0F172A',
    birdBeak: '#FBBF24',
    scoreText: '#58A6FF',
    starDot: 'rgba(88,166,255,0.15)',
  };

  // Stars (background decoration)
  var stars = [];
  for (var i = 0; i < 30; i++) {
    stars.push({
      x: Math.random() * W,
      y: Math.random() * (H - GROUND_H - 40),
      size: Math.random() * 1.5 + 0.5,
      twinkle: Math.random() * Math.PI * 2
    });
  }

  function init() {
    canvas = document.getElementById('fb-canvas');
    ctx = canvas.getContext('2d');
    document.getElementById('fb-best').textContent = best;

    // Events
    document.getElementById('fb-start-btn').addEventListener('click', function() {
      startGame();
    });

    document.addEventListener('keydown', function(e) {
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        handleFlap();
      }
    });

    canvas.addEventListener('click', function() {
      handleFlap();
    });

    canvas.addEventListener('touchstart', function(e) {
      e.preventDefault();
      handleFlap();
    });

    // Initial idle render
    requestAnimationFrame(loop);
  }

  function handleFlap() {
    if (state === STATE_IDLE) {
      startGame();
    } else if (state === STATE_PLAYING) {
      flap();
    } else if (state === STATE_DEAD) {
      restart();
    }
  }

  function startGame() {
    state = STATE_PLAYING;
    score = 0;
    bird.x = 80;
    bird.y = H / 2;
    bird.vy = 0;
    bird.angle = 0;
    pipes = [];
    frameCount = 0;
    document.getElementById('fb-score').textContent = '0';
    document.getElementById('fb-overlay').style.display = 'none';
  }

  function restart() {
    startGame();
  }

  function flap() {
    bird.vy = FLAP_POWER;
  }

  function loop() {
    update();
    draw();
    requestAnimationFrame(loop);
  }

  function update() {
    if (state !== STATE_PLAYING) return;

    frameCount++;

    // Bird physics
    bird.vy = Math.min(bird.vy + GRAVITY, TERMINAL_VEL);
    bird.y += bird.vy;
    bird.angle = Math.min(Math.max(bird.vy * 3, -30), 70);

    // Ground scroll
    groundX = (groundX - PIPE_SPEED) % 24;

    // Spawn pipes
    if (frameCount % PIPE_INTERVAL === 0) {
      var minTop = 60;
      var maxTop = H - GROUND_H - PIPE_GAP - 60;
      var topH = Math.random() * (maxTop - minTop) + minTop;
      pipes.push({
        x: W,
        topH: topH,
        scored: false
      });
    }

    // Move pipes & score
    for (var i = pipes.length - 1; i >= 0; i--) {
      pipes[i].x -= PIPE_SPEED;

      // Score
      if (!pipes[i].scored && pipes[i].x + PIPE_WIDTH < bird.x) {
        pipes[i].scored = true;
        score++;
        document.getElementById('fb-score').textContent = score;
      }

      // Remove off-screen
      if (pipes[i].x + PIPE_WIDTH < -10) {
        pipes.splice(i, 1);
      }
    }

    // Collision: ground / ceiling
    if (bird.y + bird.radius > H - GROUND_H) {
      die();
      return;
    }
    if (bird.y - bird.radius < 0) {
      bird.y = bird.radius;
      bird.vy = 0;
    }

    // Collision: pipes
    for (var i = 0; i < pipes.length; i++) {
      var p = pipes[i];
      if (bird.x + bird.radius > p.x && bird.x - bird.radius < p.x + PIPE_WIDTH) {
        if (bird.y - bird.radius < p.topH || bird.y + bird.radius > p.topH + PIPE_GAP) {
          die();
          return;
        }
      }
    }
  }

  function die() {
    state = STATE_DEAD;
    if (score > best) {
      best = score;
      localStorage.setItem('bestFlappy', best);
      document.getElementById('fb-best').textContent = best;
    }
    document.getElementById('fb-overlay-msg').textContent = 'GAME OVER';
    document.getElementById('fb-overlay-sub').textContent = 'Score: ' + score + ' | Press SPACE or Tap to retry';
    document.getElementById('fb-overlay').style.display = 'flex';
    document.getElementById('fb-start-btn').textContent = '$ retry';
  }

  function draw() {
    // Sky gradient
    var grad = ctx.createLinearGradient(0, 0, 0, H - GROUND_H);
    grad.addColorStop(0, COL.sky1);
    grad.addColorStop(1, COL.sky2);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H - GROUND_H);

    // Stars
    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];
      var alpha = 0.3 + 0.3 * Math.sin(frameCount * 0.02 + s.twinkle);
      ctx.fillStyle = 'rgba(88,166,255,' + alpha + ')';
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pipes
    for (var i = 0; i < pipes.length; i++) {
      drawPipe(pipes[i]);
    }

    // Ground
    drawGround();

    // Bird
    drawBird();

    // In-game score (big)
    if (state === STATE_PLAYING) {
      ctx.save();
      ctx.font = '700 36px ' + getComputedStyle(document.documentElement).getPropertyValue('--oc-mono').trim();
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(88,166,255,0.35)';
      ctx.fillText(score, W / 2, 60);
      ctx.restore();
    }
  }

  function drawPipe(p) {
    var capH = 18;
    var capExtra = 6;

    // Top pipe body
    var grad1 = ctx.createLinearGradient(p.x, 0, p.x + PIPE_WIDTH, 0);
    grad1.addColorStop(0, COL.pipeHighlight);
    grad1.addColorStop(0.3, COL.pipe);
    grad1.addColorStop(1, 'rgba(37,99,235,0.4)');
    ctx.fillStyle = grad1;
    ctx.fillRect(p.x, 0, PIPE_WIDTH, p.topH - capH);

    // Top pipe cap
    ctx.fillStyle = COL.pipeCap;
    ctx.fillRect(p.x - capExtra, p.topH - capH, PIPE_WIDTH + capExtra * 2, capH);
    ctx.strokeStyle = COL.pipeBorder;
    ctx.lineWidth = 1;
    ctx.strokeRect(p.x - capExtra, p.topH - capH, PIPE_WIDTH + capExtra * 2, capH);

    // Top pipe body border
    ctx.strokeStyle = COL.pipeBorder;
    ctx.lineWidth = 1;
    ctx.strokeRect(p.x, 0, PIPE_WIDTH, p.topH - capH);

    // Bottom pipe
    var bottomY = p.topH + PIPE_GAP;
    var bottomH = H - GROUND_H - bottomY;

    // Bottom pipe cap
    ctx.fillStyle = COL.pipeCap;
    ctx.fillRect(p.x - capExtra, bottomY, PIPE_WIDTH + capExtra * 2, capH);
    ctx.strokeStyle = COL.pipeBorder;
    ctx.strokeRect(p.x - capExtra, bottomY, PIPE_WIDTH + capExtra * 2, capH);

    // Bottom pipe body
    var grad2 = ctx.createLinearGradient(p.x, 0, p.x + PIPE_WIDTH, 0);
    grad2.addColorStop(0, COL.pipeHighlight);
    grad2.addColorStop(0.3, COL.pipe);
    grad2.addColorStop(1, 'rgba(37,99,235,0.4)');
    ctx.fillStyle = grad2;
    ctx.fillRect(p.x, bottomY + capH, PIPE_WIDTH, bottomH - capH);
    ctx.strokeStyle = COL.pipeBorder;
    ctx.strokeRect(p.x, bottomY + capH, PIPE_WIDTH, bottomH - capH);
  }

  function drawGround() {
    // Ground base
    ctx.fillStyle = COL.ground1;
    ctx.fillRect(0, H - GROUND_H, W, GROUND_H);

    // Ground top line
    ctx.fillStyle = COL.ground2;
    ctx.fillRect(0, H - GROUND_H, W, 3);

    // Ground grid lines (scrolling)
    ctx.strokeStyle = COL.groundLine;
    ctx.lineWidth = 1;
    for (var x = groundX; x < W; x += 24) {
      ctx.beginPath();
      ctx.moveTo(x, H - GROUND_H + 3);
      ctx.lineTo(x - 12, H);
      ctx.stroke();
    }
    for (var y = H - GROUND_H + 15; y < H; y += 15) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
  }

  function drawBird() {
    ctx.save();
    ctx.translate(bird.x, bird.y);
    ctx.rotate(bird.angle * Math.PI / 180);

    var r = bird.radius;

    // Bird body glow
    ctx.shadowColor = 'rgba(88,166,255,0.4)';
    ctx.shadowBlur = 12;

    // Bird body (circle placeholder)
    ctx.fillStyle = COL.bird;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Bird border
    ctx.strokeStyle = COL.birdBorder;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Wing (simple arc)
    var wingFlap = Math.sin(frameCount * 0.3) * 4;
    ctx.fillStyle = COL.birdWing;
    ctx.beginPath();
    ctx.ellipse(-4, 2 + wingFlap, 8, 5, -0.2, 0, Math.PI * 2);
    ctx.fill();

    // Eye
    ctx.fillStyle = COL.birdEye;
    ctx.beginPath();
    ctx.arc(5, -3, 3, 0, Math.PI * 2);
    ctx.fill();

    // Eye highlight
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(6, -4, 1.2, 0, Math.PI * 2);
    ctx.fill();

    // Beak
    ctx.fillStyle = COL.birdBeak;
    ctx.beginPath();
    ctx.moveTo(r - 2, -2);
    ctx.lineTo(r + 7, 1);
    ctx.lineTo(r - 2, 4);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { restart: restart };
})();
</script>
