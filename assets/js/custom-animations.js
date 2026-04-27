/* ============================================
   Custom Animations - Terminal/Geek Style
   ============================================ */

var CustomAnimations = {
  init: function () {
    CustomAnimations.initTerminalEffects();
    CustomAnimations.initCardInteractions();
    CustomAnimations.initCursorBlink();
  },

  // Terminal-style typing and cursor effects
  initTerminalEffects: function () {
    // Add hover command preview to sidebar labels
    document.querySelectorAll('.oc-sidebar-label').forEach(function (label) {
      label.addEventListener('mouseenter', function () {
        var original = label.textContent;
        label.setAttribute('data-original', original);
        label.textContent = '$ cat ' + original.replace('// ', '') + '/';
      });
      label.addEventListener('mouseleave', function () {
        label.textContent = label.getAttribute('data-original');
      });
    });
  },

  // Card border glow on hover
  initCardInteractions: function () {
    document.querySelectorAll('.oc-card').forEach(function (card) {
      card.addEventListener('mouseenter', function () {
        card.style.borderColor = 'var(--oc-blue)';
      });
      card.addEventListener('mouseleave', function () {
        card.style.borderColor = '';
      });
    });
  },

  // Blinking cursor at bottom
  initCursorBlink: function () {
    var cursors = document.querySelectorAll('.oc-cursor');
    cursors.forEach(function (cursor) {
      cursor.style.animation = 'oc-blink 1s step-end infinite';
    });
  }
};

document.addEventListener('DOMContentLoaded', CustomAnimations.init);
