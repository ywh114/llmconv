/**
 * Ara VN UI Components
 *
 * Title screen, settings panel, and high-level UI wiring.
 */

(function () {
  'use strict';

  const $title = document.getElementById('vn-title');
  const $settings = document.getElementById('vn-settings');
  const $history = document.getElementById('vn-history');
  const $keybinds = document.getElementById('vn-keybinds');
  const $keybindsHint = document.getElementById('vn-keybinds-hint');

  // -------------------------------------------------------------------
  // Title screen
  // -------------------------------------------------------------------
  document.getElementById('btn-start').addEventListener('click', () => {
    $title.classList.add('vn-hidden');
    $settings.classList.remove('vn-visible');
    $keybinds.classList.remove('vn-visible');
    window.VN.start();
  });

  document.getElementById('btn-settings').addEventListener('click', () => {
    $settings.classList.add('vn-visible');
  });

  document.getElementById('btn-keybinds').addEventListener('click', () => {
    $keybinds.classList.add('vn-visible');
  });

  // -------------------------------------------------------------------
  // Settings panel
  // -------------------------------------------------------------------
  document.getElementById('btn-settings-close').addEventListener('click', () => {
    $settings.classList.remove('vn-visible');
  });

  // -------------------------------------------------------------------
  // Keybinds panel
  // -------------------------------------------------------------------
  document.getElementById('btn-keybinds-close').addEventListener('click', () => {
    $keybinds.classList.remove('vn-visible');
  });

  const $textSpeed = document.getElementById('setting-text-speed');
  $textSpeed.addEventListener('input', (e) => {
    // Invert: slider high = fast speed = low delay
    const val = parseInt(e.target.value, 10);
    const delay = Math.max(5, 105 - val);
    window.VN.setTextSpeed(delay);
  });

  const $autoDelay = document.getElementById('setting-auto-delay');
  $autoDelay.addEventListener('input', (e) => {
    window.VN.setAutoDelay(e.target.value);
  });

  // -------------------------------------------------------------------
  // History overlay — click outside to close
  // -------------------------------------------------------------------
  $history.addEventListener('click', (e) => {
    if (e.target === $history) {
      $history.classList.remove('vn-visible');
    }
  });

  // -------------------------------------------------------------------
  // Debug helpers (optional keyboard shortcuts)
  // -------------------------------------------------------------------
  document.addEventListener('keydown', (e) => {
    // Ignore shortcuts when typing in an input field
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
      return;
    }
    // S toggles settings
    if (e.code === 'KeyS') {
      $settings.classList.toggle('vn-visible');
      $keybinds.classList.remove('vn-visible');
      return;
    }
    // K toggles keybinds reference
    if (e.code === 'KeyK') {
      $keybinds.classList.toggle('vn-visible');
      $settings.classList.remove('vn-visible');
      return;
    }
    // Escape closes overlays
    if (e.code === 'Escape') {
      $settings.classList.remove('vn-visible');
      $history.classList.remove('vn-visible');
      $keybinds.classList.remove('vn-visible');
    }
  });
})();
