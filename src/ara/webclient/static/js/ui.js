/**
 * Ara VN UI Components
 *
 * Title screen, settings panel, saveload panel, debug panel, and high-level UI wiring.
 */

(function () {
  'use strict';

  const $title = document.getElementById('vn-title');
  const $settings = document.getElementById('vn-settings');
  const $history = document.getElementById('vn-history');
  const $keybinds = document.getElementById('vn-keybinds');
  const $keybindsHint = document.getElementById('vn-keybinds-hint');
  const $saveload = document.getElementById('vn-saveload');
  const $saveloadGrid = document.getElementById('vn-saveload-grid');
  const $debug = document.getElementById('vn-debug');
  const $debugOutput = document.getElementById('vn-debug-output');
  const $system = document.getElementById('vn-system');
  const $systemContent = document.getElementById('vn-system-content');
  let currentSystemState = {};

  // -------------------------------------------------------------------
  // System page renderer
  // -------------------------------------------------------------------
  function renderSection(section) {
    const type = section.type;
    const items = section.items || [];
    const el = document.createElement('div');
    el.className = 'vn-system-section';

    if (type === 'bars') {
      el.innerHTML = '<h3>Status</h3>';
      items.forEach(item => {
        const max = item.max || 100;
        const pct = Math.max(0, Math.min(100, (Number(item.value) / max) * 100));
        const color = item.color ? ` style="background:${escapeHtml(item.color)}"` : '';
        el.innerHTML += `
          <div class="vn-system-bar">
            <label>${escapeHtml(item.label || '')}</label>
            <div class="vn-system-bar-track">
              <div class="vn-system-bar-fill"${color} style="width:${pct}%"></div>
            </div>
          </div>`;
      });
      return el;
    }

    const title = type === 'inventory' ? 'Inventory' :
                  type === 'skills' ? 'Skills' :
                  section.title || type;
    el.innerHTML = `<h3>${escapeHtml(title)}</h3>`;
    const ul = document.createElement('ul');
    ul.className = 'vn-system-list';
    items.forEach(item => {
      const li = document.createElement('li');
      li.className = 'vn-system-item';

      // First-class Item objects carry structured fields and are rendered distinctly.
      if (typeof item === 'object' && item.id) {
        li.classList.add('vn-system-item-important');

        const row = document.createElement('div');
        row.className = 'vn-system-item-row';

        if (item.icon) {
          const icon = document.createElement('img');
          icon.className = 'vn-system-item-icon';
          icon.src = `/assets/${escapeHtml(String(item.icon))}.png`;
          icon.alt = '';
          icon.onerror = () => { icon.style.display = 'none'; };
          row.appendChild(icon);
        }

        const name = document.createElement('span');
        name.className = 'vn-system-item-name';
        name.textContent = item.name || item.id;
        row.appendChild(name);

        if (item.quantity && Number(item.quantity) > 1) {
          const qty = document.createElement('span');
          qty.className = 'vn-system-item-qty';
          qty.textContent = `×${item.quantity}`;
          row.appendChild(qty);
        }

        li.appendChild(row);

        if (item.description) {
          li.classList.add('vn-system-item-has-desc');
          li.title = item.description;
          const descEl = document.createElement('div');
          descEl.className = 'vn-system-item-desc';
          descEl.textContent = item.description;
          if (Array.isArray(item.tags) && item.tags.length) {
            const tagsEl = document.createElement('div');
            tagsEl.className = 'vn-system-item-tags';
            tagsEl.textContent = item.tags.join(', ');
            descEl.appendChild(tagsEl);
          }
          li.appendChild(descEl);
          li.addEventListener('click', () => {
            li.classList.toggle('vn-system-item-expanded');
          });
        }
      } else {
        const text = typeof item === 'string' ? item : (item.label || item.name || String(item));
        const desc = typeof item === 'object' ? (item.description || '') : '';
        li.textContent = text;
        if (desc) {
          li.classList.add('vn-system-item-has-desc');
          li.title = desc;
          const descEl = document.createElement('div');
          descEl.className = 'vn-system-item-desc';
          descEl.textContent = desc;
          li.appendChild(descEl);
          li.addEventListener('click', () => {
            li.classList.toggle('vn-system-item-expanded');
          });
        }
      }

      ul.appendChild(li);
    });
    el.appendChild(ul);
    return el;
  }

  function renderSystem() {
    if (!$systemContent) return;
    $systemContent.innerHTML = '<div style="color:var(--vn-gold)">Loading…</div>';

    fetch('/state')
      .then(resp => {
        if (!resp.ok) throw new Error('Failed to fetch state');
        return resp.json();
      })
      .then(data => {
        currentSystemState = (data.engine && data.engine.player_status) || {};
        $systemContent.innerHTML = '';

        const title = currentSystemState.title;
        if (title) {
          const heading = document.createElement('h2');
          heading.className = 'vn-system-title';
          heading.textContent = title;
          $systemContent.appendChild(heading);
        }

        const sections = currentSystemState.sections || [];
        if (sections.length) {
          sections.forEach(section => {
            $systemContent.appendChild(renderSection(section));
          });
        }

        // Backward-compatible flat-key rendering.
        const legacyKeys = ['bars', 'inventory', 'skills'];
        const hasLegacy = legacyKeys.some(k => currentSystemState[k] !== undefined);
        if (hasLegacy) {
          if (currentSystemState.bars) {
            $systemContent.appendChild(renderSection({
              type: 'bars',
              items: Object.entries(currentSystemState.bars).map(([label, value]) => ({ label, value })),
            }));
          }
          ['inventory', 'skills'].forEach(key => {
            if (currentSystemState[key]) {
              $systemContent.appendChild(renderSection({ type: key, items: currentSystemState[key] }));
            }
          });
        }

        if (!$systemContent.children.length) {
          $systemContent.innerHTML = '<div style="color:var(--vn-text-muted)">No system data available.</div>';
        }
      })
      .catch(err => {
        $systemContent.innerHTML = `<div style="color:#f55">Error: ${escapeHtml(err.message)}</div>`;
      });
  }

  function applySystemChanges(changes) {
    if (!changes || typeof changes !== 'object') return;
    if (changes.sections) {
      currentSystemState = changes;
    } else {
      Object.assign(currentSystemState, changes);
    }
    if ($system && $system.classList.contains('vn-visible')) {
      renderSystem();
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.applySystemChanges = applySystemChanges;

  // -------------------------------------------------------------------
  // Story selection
  // -------------------------------------------------------------------
  async function populateStories() {
    const container = document.getElementById('vn-story-select');
    if (!container) return;
    try {
      const resp = await fetch('/stories');
      if (!resp.ok) return;
      const data = await resp.json();
      const stories = data.stories || [];
      if (stories.length <= 1) {
        container.style.display = 'none';
        return;
      }
      container.innerHTML = '';
      const select = document.createElement('select');
      select.id = 'vn-story-picker';
      select.className = 'vn-story-picker';
      stories.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.title || s.id;
        select.appendChild(opt);
      });
      container.appendChild(select);
    } catch (err) {
      console.warn('Could not load stories:', err);
    }
  }
  populateStories();

  // -------------------------------------------------------------------
  // Title screen
  // -------------------------------------------------------------------
  document.getElementById('btn-start').addEventListener('click', () => {
    const picker = document.getElementById('vn-story-picker');
    const storyId = picker ? picker.value : null;
    $title.classList.add('vn-hidden');
    $settings.classList.remove('vn-visible');
    $keybinds.classList.remove('vn-visible');
    $saveload.classList.remove('vn-visible');
    window.VN.start(null, storyId);
  });

  document.getElementById('btn-load').addEventListener('click', () => {
    console.log('[UI] Load button clicked');
    openSaveload().catch(err => console.error('[UI] openSaveload failed:', err));
  });

  // -------------------------------------------------------------------
  // Save/Load panel
  // -------------------------------------------------------------------

  async function openSaveload() {
    $saveload.classList.add('vn-visible');
    const onTitle = !$title.classList.contains('vn-hidden');
    $saveloadGrid.innerHTML = '<div style="color:var(--vn-gold)">Loading…</div>';
    try {
      const data = await window.VN.listSaves();
      const saves = data.saves || [];
      const saveMap = new Map(saves.map(s => [s.slot, s]));
      $saveloadGrid.innerHTML = '';
      for (let slot = 1; slot <= 20; slot++) {
        const cell = document.createElement('div');
        cell.className = 'vn-saveload-cell';
        const saved = saveMap.get(slot);

        const slotInfo = saved
          ? `<div class="vn-saveload-slot">#${slot}</div>
             <div class="vn-saveload-scene">${saved.scene_id || '?'}</div>
             <div class="vn-saveload-time">${saved.timestamp ? saved.timestamp.slice(0, 16).replace('T', ' ') : ''}</div>`
          : `<div class="vn-saveload-slot empty">#${slot}</div><div class="vn-saveload-scene">Empty</div>`;

        const overlay = document.createElement('div');
        overlay.className = 'vn-saveload-overlay';

        const btnSave = document.createElement('button');
        btnSave.className = 'vn-title-btn';
        btnSave.textContent = 'Save';
        btnSave.disabled = onTitle;
        if (!onTitle) {
          btnSave.onclick = (e) => {
            e.stopPropagation();
            window.VN.save(slot).then(() => openSaveload()).catch(err => {
              alert('Save failed: ' + err.message);
            });
          };
        }

        const btnLoad = document.createElement('button');
        btnLoad.className = 'vn-title-btn';
        btnLoad.textContent = 'Load';
        btnLoad.disabled = !saved;
        if (saved) {
          btnLoad.onclick = (e) => {
            e.stopPropagation();
            $saveload.classList.remove('vn-visible');
            $title.classList.add('vn-hidden');
            window.VN.load(slot).catch(err => {
              alert('Load failed: ' + err.message);
            });
          };
        }

        const btnDelete = document.createElement('button');
        btnDelete.className = 'vn-title-btn';
        btnDelete.textContent = 'Delete';
        btnDelete.disabled = !saved;
        if (saved) {
          btnDelete.onclick = (e) => {
            e.stopPropagation();
            if (confirm(`Delete save slot ${slot}?`)) {
              window.VN.delete(slot).then(() => openSaveload()).catch(err => {
                alert('Delete failed: ' + err.message);
              });
            }
          };
        }

        overlay.appendChild(btnSave);
        overlay.appendChild(btnLoad);
        overlay.appendChild(btnDelete);

        cell.innerHTML = slotInfo;
        cell.appendChild(overlay);
        $saveloadGrid.appendChild(cell);
      }
    } catch (err) {
      $saveloadGrid.innerHTML = `<div style="color:#f55">Error: ${err.message}</div>`;
    }
  }

  document.getElementById('btn-saveload-close').addEventListener('click', () => {
    $saveload.classList.remove('vn-visible');
  });

  // -------------------------------------------------------------------
  // Debug panel
  // -------------------------------------------------------------------
  document.getElementById('btn-debug-run').addEventListener('click', async () => {
    const cmd = document.getElementById('vn-debug-cmd').value.trim();
    const argsStr = document.getElementById('vn-debug-args').value.trim();
    const args = argsStr ? argsStr.split(/\s+/) : [];
    if (!cmd) return;
    $debugOutput.textContent = 'Running…';
    try {
      const result = await window.VN.debug(cmd, args);
      $debugOutput.textContent = JSON.stringify(result, null, 2);
    } catch (err) {
      $debugOutput.textContent = 'Error: ' + err.message;
    }
  });

  document.getElementById('btn-debug-close').addEventListener('click', () => {
    $debug.classList.remove('vn-visible');
  });

  // -------------------------------------------------------------------
  // Settings panel
  // -------------------------------------------------------------------
  document.getElementById('btn-settings').addEventListener('click', () => {
    $settings.classList.add('vn-visible');
  });

  document.getElementById('btn-settings-close').addEventListener('click', () => {
    $settings.classList.remove('vn-visible');
  });

  // -------------------------------------------------------------------
  // Keybinds panel
  // -------------------------------------------------------------------
  document.getElementById('btn-keybinds').addEventListener('click', () => {
    $keybinds.classList.add('vn-visible');
  });

  document.getElementById('btn-keybinds-close').addEventListener('click', () => {
    $keybinds.classList.remove('vn-visible');
  });

  if (document.getElementById('btn-system-close')) {
    document.getElementById('btn-system-close').addEventListener('click', () => {
      $system.classList.remove('vn-visible');
    });
  }

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
      $saveload.classList.remove('vn-visible');
      $debug.classList.remove('vn-visible');
      return;
    }
    // K toggles keybinds reference
    if (e.code === 'KeyK') {
      $keybinds.classList.toggle('vn-visible');
      $settings.classList.remove('vn-visible');
      $saveload.classList.remove('vn-visible');
      $debug.classList.remove('vn-visible');
      return;
    }
    // D toggles debug panel
    if (e.code === 'KeyD') {
      $debug.classList.toggle('vn-visible');
      $settings.classList.remove('vn-visible');
      $keybinds.classList.remove('vn-visible');
      $saveload.classList.remove('vn-visible');
      return;
    }
    // L toggles saveload panel (load mode)
    if (e.code === 'KeyL') {
      $saveload.classList.toggle('vn-visible');
      $settings.classList.remove('vn-visible');
      $keybinds.classList.remove('vn-visible');
      $debug.classList.remove('vn-visible');
      $system.classList.remove('vn-visible');
      if ($saveload.classList.contains('vn-visible')) {
        openSaveload();
      }
      return;
    }
    // E toggles the system page
    if (e.code === 'KeyE') {
      $system.classList.toggle('vn-visible');
      $settings.classList.remove('vn-visible');
      $keybinds.classList.remove('vn-visible');
      $saveload.classList.remove('vn-visible');
      $debug.classList.remove('vn-visible');
      if ($system.classList.contains('vn-visible')) {
        renderSystem();
      }
      return;
    }
    // Quick save with F5
    if (e.code === 'F5') {
      e.preventDefault();
      window.VN.save(1).then(() => {
        console.log('Quick saved to slot 1');
      }).catch(err => {
        console.error('Quick save failed:', err);
        alert('Quick save failed: ' + err.message);
      });
      return;
    }
    // Escape closes overlays
    if (e.code === 'Escape') {
      $settings.classList.remove('vn-visible');
      $history.classList.remove('vn-visible');
      $keybinds.classList.remove('vn-visible');
      $saveload.classList.remove('vn-visible');
      $debug.classList.remove('vn-visible');
      $system.classList.remove('vn-visible');
    }
  });

  // -------------------------------------------------------------------
  // In-game control bar (under the textbox)
  // -------------------------------------------------------------------
  const $controls = document.getElementById('vn-controls');
  if ($controls) {
    $controls.addEventListener('click', (e) => {
      const btn = e.target.closest('.vn-control-btn');
      if (!btn) return;
      const action = btn.dataset.action;
      switch (action) {
        case 'save':
          window.VN.save(1).catch(err => alert('Save failed: ' + err.message));
          break;
        case 'load':
          openSaveload();
          break;
        case 'history':
          window.VN.toggleHistory();
          break;
        case 'auto': {
          const on = window.VN.toggleAuto();
          btn.classList.toggle('active', on);
          break;
        }
        case 'settings':
          $settings.classList.add('vn-visible');
          break;
      }
    });
  }
})();
