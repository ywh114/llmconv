/**
 * Ara VN Frontend — simple polling GUI for aractl.
 *
 * Fetches one event at a time via /step, renders it, waits for click,
 * then fetches the next.  No SSE, no batching.
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------------
     DOM
     ------------------------------------------------------------------ */
  const $bg = document.getElementById('vn-background');
  const $sprites = document.getElementById('vn-sprites');
  const $name = document.getElementById('vn-name');
  const $text = document.getElementById('vn-text');
  const $choices = document.getElementById('vn-choices');
  const $history = document.getElementById('vn-history');
  const $historyList = document.getElementById('vn-history-list');

  /* ------------------------------------------------------------------
     State
     ------------------------------------------------------------------ */
  const STATE = {
    pool: [],
    here: [],
    narrator: null,
    history: [],
    textSpeed: 30,
    autoDelay: 2500,
  };

  let _running = false;
  let _loopActive = false;
  let _typingTimer = null;
  let _fullText = '';
  let _pendingClick = null;
  let _autoMode = false;

  /* ------------------------------------------------------------------
     Helpers
     ------------------------------------------------------------------ */
  function assetUrl(type, name) {
    return '/assets/' + type + '/' + name;
  }

  function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  /* ------------------------------------------------------------------
     Background
     ------------------------------------------------------------------ */
  function setBackground(locName) {
    if (!locName) return;
    $bg.style.backgroundImage = `url('${assetUrl('bg', locName + '.png')}')`;
  }

  /* ------------------------------------------------------------------
     Sprites
     ------------------------------------------------------------------ */
  function clearSprites() {
    $sprites.innerHTML = '';
  }

  function applyCropStyles(wrapper, img) {
    const crop = wrapper._crop;
    if (!crop || !img.naturalWidth) return;

    const [x1, y1] = crop.topleft;
    const [x2, y2] = crop.bottomright;
    const cropW = x2 - x1;
    const cropH = y2 - y1;
    if (cropW <= 0 || cropH <= 0) return;

    const fullW = img.naturalWidth;
    const fullH = img.naturalHeight;
    const renderH = wrapper.clientHeight || window.innerHeight * 0.75;
    const scale = renderH / cropH;

    img.style.width = `${fullW * scale}px`;
    img.style.height = `${fullH * scale}px`;

    const position = wrapper.dataset.position || 'center';
    let offsetX = 0;
    if (position === 'left') {
      offsetX = -x1 * scale;
    } else if (position === 'center') {
      offsetX = -(x1 + cropW / 2) * scale;
    } else if (position === 'right') {
      offsetX = -(x1 + cropW) * scale;
    }

    const lift = wrapper.classList.contains('vn-speaking') ? -8 : 0;
    // Align the crop region's bottom with the wrapper's bottom.
    // The full image extends naturally; nothing is clipped.
    img.style.transform = `translate(${offsetX}px, ${(fullH - y2) * scale + lift}px)`;
  }

  function updateSprites(hereChars, speakerName) {
    // Remove sprites that left
    Array.from($sprites.children).forEach(el => {
      if (!hereChars.find(c => c.name === el.dataset.name)) {
        el.classList.add('vn-hidden');
        setTimeout(() => el.remove(), 600);
      }
    });

    const visibleChars = hereChars.filter(c => c.name !== STATE.narrator);
    visibleChars.forEach((c, i) => {
      let el = $sprites.querySelector(`[data-name="${c.name}"]`);
      const isSpeaking = c.name === speakerName;
      const position = visibleChars.length === 1
        ? 'center'
        : i === 0 ? 'left' : i === visibleChars.length - 1 ? 'right' : 'center';

      const spriteName = c.current_sprite || 'neutral';
      const needsRebuild = el && el.dataset.sprite !== spriteName;
      if (needsRebuild) {
        el.remove();
        el = null;
      }

      if (!el) {
        const isAnon = c.importance === 'ANONYMOUS';
        const spriteId = c.sprite || '';
        if (isAnon && !spriteId) return;

        const url = isAnon && spriteId
          ? assetUrl('cc', 'anonymous/' + spriteId + '.png')
          : assetUrl('cc', c.name + '/' + spriteName + '.png');

        // Check for a custom crop region in meta.toml (keyed by sprite name)
        const crop = c.crops && c.crops[spriteName];

        if (crop && crop.topleft && crop.bottomright) {
          // Cropped sprite: wrapper positions the crop region;
          // the full image remains visible (no overflow:hidden).
          const wrapper = document.createElement('div');
          wrapper.className = 'vn-sprite-crop vn-hidden';
          wrapper.dataset.name = c.name;
          wrapper._crop = crop;
          $sprites.appendChild(wrapper);

          const img = document.createElement('img');
          img.alt = c.name;
          wrapper.appendChild(img);

          img.onload = () => {
            applyCropStyles(wrapper, img);
            wrapper.classList.remove('vn-hidden');
          };
          img.onerror = () => {
            if (!isAnon) img.src = assetUrl('cc', '柴郡/neutral.png');
            wrapper.classList.remove('vn-hidden');
          };
          img.src = url;
          el = wrapper;
          el.dataset.sprite = spriteName;
        } else {
          // Regular uncropped sprite
          el = document.createElement('img');
          el.className = 'vn-sprite vn-hidden';
          el.dataset.name = c.name;
          el.alt = c.name;
          $sprites.appendChild(el);

          el.src = url;
          const tempImg = new Image();
          tempImg.onload = () => el.classList.remove('vn-hidden');
          tempImg.onerror = () => {
            if (!isAnon) el.src = assetUrl('cc', '柴郡/neutral.png');
            el.classList.remove('vn-hidden');
          };
          tempImg.src = url;
          el.dataset.sprite = spriteName;
        }
      }

      el.dataset.position = position;
      el.classList.toggle('vn-speaking', isSpeaking);
      el.classList.remove('vn-hidden');

      // Re-apply crop transform when position or speaking state changes.
      if (el._crop) {
        const img = el.querySelector('img');
        if (img) applyCropStyles(el, img);
      }
    });

    // Bring the speaking sprite to the front (last in DOM order).
    visibleChars.forEach(c => {
      if (c.name === speakerName) {
        const el = $sprites.querySelector(`[data-name="${c.name}"]`);
        if (el) $sprites.appendChild(el);
      }
    });
  }

  /* ------------------------------------------------------------------
     Text paging
     ------------------------------------------------------------------ */
  function splitIntoPages(text) {
    const style = getComputedStyle($text);
    const maxHeight = parseFloat(style.maxHeight);
    if (!maxHeight || !text) return [text];

    const measurer = document.createElement('div');
    measurer.style.position = 'absolute';
    measurer.style.visibility = 'hidden';
    measurer.style.left = '-9999px';
    measurer.style.width = $text.clientWidth + 'px';
    measurer.style.font = style.font;
    measurer.style.lineHeight = style.lineHeight;
    measurer.style.whiteSpace = 'pre-wrap';
    measurer.style.wordBreak = 'break-word';
    document.body.appendChild(measurer);

    const pages = [];
    let remaining = text;

    while (remaining.length > 0) {
      let low = 1, high = remaining.length, best = 1;
      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        measurer.textContent = remaining.slice(0, mid);
        if (measurer.offsetHeight <= maxHeight) {
          best = mid;
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }

      // Prefer breaking at a word boundary
      let split = best;
      if (split < remaining.length) {
        const before = remaining.lastIndexOf(' ', split - 1);
        if (before > 0 && split - before < 25) {
          split = before + 1;
        }
      }
      if (split < 1) split = 1;

      pages.push(remaining.slice(0, split));
      remaining = remaining.slice(split);
    }

    document.body.removeChild(measurer);
    return pages.length ? pages : [text];
  }

  /* ------------------------------------------------------------------
     Text typing
     ------------------------------------------------------------------ */
  async function typeText(speaker, text) {
    const pages = splitIntoPages(text);
    $name.textContent = speaker || '';

    for (let p = 0; p < pages.length; p++) {
      _fullText = pages[p];
      $text.innerHTML = '<span class="vn-cursor"></span>';
      let i = 0;
      const chars = pages[p].split('');

      await new Promise(resolve => {
        function tick() {
          if (i >= chars.length) {
            _typingTimer = null;
            resolve();
            return;
          }
          const span = document.createElement('span');
          span.textContent = chars[i];
          $text.insertBefore(span, $text.lastElementChild);
          i++;
          _typingTimer = setTimeout(tick, STATE.textSpeed);
        }
        _typingTimer = setTimeout(tick, STATE.textSpeed);
      });

      // Wait for click before next page (not after the last page).
      if (p < pages.length - 1) {
        await waitForClick();
      }
    }
  }

  function skipTyping() {
    if (_typingTimer) {
      clearTimeout(_typingTimer);
      _typingTimer = null;
    }
    if (_fullText) {
      $text.textContent = _fullText;
      $text.innerHTML += '<span class="vn-cursor"></span>';
    }
  }

  function isTyping() {
    return _typingTimer !== null;
  }

  /* ------------------------------------------------------------------
     History
     ------------------------------------------------------------------ */
  function addToHistory(speaker, text) {
    STATE.history.push({ speaker, text });
    const entry = document.createElement('div');
    entry.className = 'vn-history-entry';
    entry.innerHTML = `
      <div class="vn-history-name">${speaker || 'Narrator'}</div>
      <div class="vn-history-text">${text}</div>
    `;
    $historyList.appendChild(entry);
  }

  /* ------------------------------------------------------------------
     Choices / Input
     ------------------------------------------------------------------ */
  function showChoices(suggestions) {
    $choices.innerHTML = '';
    $choices.classList.add('vn-visible');
    suggestions.forEach(text => {
      const btn = document.createElement('button');
      btn.className = 'vn-choice';
      btn.textContent = text;
      btn.onclick = () => generateAndSubmit(text);
      $choices.appendChild(btn);
    });

    const row = document.createElement('div');
    row.className = 'vn-custom-input-row';
    const input = document.createElement('input');
    input.className = 'vn-custom-input';
    input.type = 'text';
    input.placeholder = 'Or type your own response...';
    input.addEventListener('keydown', (e) => {
      if (e.code === 'Enter') {
        e.preventDefault();
        const text = input.value.trim();
        if (text) submitCustom(text);
      }
    });
    const send = document.createElement('button');
    send.className = 'vn-custom-send';
    send.textContent = 'Send';
    send.onclick = () => {
      const text = input.value.trim();
      if (text) submitCustom(text);
    };
    row.appendChild(input);
    row.appendChild(send);
    $choices.appendChild(row);
    input.focus();
  }

  function hideChoices() {
    $choices.classList.remove('vn-visible');
    $choices.innerHTML = '';
  }

  /* ------------------------------------------------------------------
     Event processing
     ------------------------------------------------------------------ */
  function applyEnterExit(enter, exit) {
    (enter || []).forEach(name => {
      if (!STATE.here.find(c => c.name === name)) {
        const char = STATE.pool.find(c => c.name === name);
        if (char) STATE.here.push(char);
      }
    });
    (exit || []).forEach(name => {
      const idx = STATE.here.findIndex(c => c.name === name);
      if (idx !== -1) STATE.here.splice(idx, 1);
    });
  }

  async function processEvent(ev) {
    if (ev.type === 'scene_loaded') {
      STATE.narrator = ev.narrator || null;
      STATE.pool = ev.characters || [];
      const starting = new Set(ev.starting_characters || []);
      STATE.here = (ev.characters || []).filter(c => starting.has(c.name));
      setBackground(ev.location);
      clearSprites();
      updateSprites(STATE.here, null);
      $name.textContent = '';
      $text.innerHTML = '';
      return 'continue';
    }

    if (ev.type === 'turn') {
      applyEnterExit(ev.enter, ev.exit);
      if (ev.sprite_changes) {
        Object.entries(ev.sprite_changes).forEach(([name, sprite]) => {
          const char = STATE.here.find(c => c.name === name);
          if (char) char.current_sprite = sprite;
        });
      }
      updateSprites(STATE.here, ev.speaker || null);

      const output = ev.output || '';
      if (!output) return 'continue';

      addToHistory(ev.speaker || null, output);
      await typeText(ev.speaker || null, output);
      return 'wait';
    }

    if (ev.type === 'needs_player_input') {
      applyEnterExit(ev.enter, ev.exit);
      if (ev.sprite_changes) {
        Object.entries(ev.sprite_changes).forEach(([name, sprite]) => {
          const char = STATE.here.find(c => c.name === name);
          if (char) char.current_sprite = sprite;
        });
      }
      updateSprites(STATE.here, ev.speaker || null);
      showChoices(ev.suggestions || []);
      return 'input';
    }

    if (ev.type === 'scene_ended') {
      $name.textContent = '';
      $text.textContent = '— Scene End —';
      hideChoices();
      return 'continue';
    }

    if (ev.type === 'story_complete') {
      $name.textContent = '';
      $text.textContent = '— The End —';
      hideChoices();
      return 'done';
    }

    return 'continue';
  }

  /* ------------------------------------------------------------------
     Click-to-advance
     ------------------------------------------------------------------ */
  function waitForClick() {
    return new Promise(resolve => {
      let autoTimer = null;

      function finish() {
        if (autoTimer) {
          clearTimeout(autoTimer);
          autoTimer = null;
        }
        cleanup();
        resolve();
      }

      function onClick(e) {
        if (e.target.closest('.vn-choice') ||
            e.target.closest('.vn-title-btn') ||
            e.target.closest('#vn-settings') ||
            e.target.closest('#vn-history') ||
            e.target.closest('#vn-keybinds')) {
          return;
        }
        if (isTyping()) {
          skipTyping();
          return;
        }
        finish();
      }
      function onKey(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
          return;
        }
        if (e.code === 'Space') {
          e.preventDefault();
          if (isTyping()) {
            skipTyping();
            return;
          }
          finish();
        }
      }
      function cleanup() {
        document.removeEventListener('click', onClick);
        document.removeEventListener('keydown', onKey);
      }
      document.addEventListener('click', onClick);
      document.addEventListener('keydown', onKey);

      if (_autoMode) {
        autoTimer = setTimeout(finish, STATE.autoDelay);
      }
    });
  }

  /* ------------------------------------------------------------------
     Main loop
     ------------------------------------------------------------------ */
  async function gameLoop() {
    if (_loopActive) return;
    _loopActive = true;
    while (_running) {
      try {
        // Loading indicator while waiting for the server.
        $name.textContent = '';
        $text.innerHTML = '<span style="color:var(--vn-gold)">…</span>';

        const resp = await fetch('/step', { method: 'POST' });
        if (!resp.ok) {
          console.error('/step failed', await resp.text());
          await sleep(1000);
          continue;
        }
        const ev = await resp.json();
        console.log('Step event:', ev);

        const result = await processEvent(ev);
        if (result === 'wait') {
          await waitForClick();
        } else if (result === 'input') {
          break;
        } else if (result === 'done') {
          _running = false;
          break;
        }
      } catch (err) {
        console.error('Game loop error', err);
        await sleep(1000);
      }
    }
    _loopActive = false;
  }

  /* ------------------------------------------------------------------
     Network
     ------------------------------------------------------------------ */
  async function postStart(sceneId) {
    const body = sceneId ? JSON.stringify({ scene_id: sceneId }) : '{}';
    const resp = await fetch('/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    if (!resp.ok) throw new Error('Start failed');
    return resp.json();
  }

  async function postInput(text) {
    const resp = await fetch('/input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) throw new Error('Input failed');
    return resp.json();
  }

  async function postGenerate(suggestion) {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suggestion }),
    });
    if (!resp.ok) throw new Error('Generate failed');
    return resp.json();
  }

  /* ------------------------------------------------------------------
     Input handlers
     ------------------------------------------------------------------ */
  async function submitInput(text) {
    hideChoices();
    addToHistory('Player', text);
    await typeText('Player', text);
    await waitForClick();
    await postInput(text);
    gameLoop();
  }

  async function generateAndSubmit(suggestion) {
    hideChoices();
    const data = await postGenerate(suggestion);
    const text = data.text || suggestion;
    addToHistory('Player', text);
    await typeText('Player', text);
    await waitForClick();
    await postInput(text);
    gameLoop();
  }

  async function submitCustom(text) {
    hideChoices();
    addToHistory('Player', text);
    await typeText('Player', text);
    await waitForClick();
    await postInput(text);
    gameLoop();
  }

  /* ------------------------------------------------------------------
     Public API
     ------------------------------------------------------------------ */
  window.VN = {
    async start(sceneId) {
      _running = true;
      await postStart(sceneId);
      gameLoop();
    },
    setTextSpeed(v) {
      STATE.textSpeed = parseInt(v, 10);
    },
    setAutoDelay(v) {
      STATE.autoDelay = parseInt(v, 10);
    },
    toggleAuto() {
      _autoMode = !_autoMode;
      const badge = document.getElementById('vn-auto-badge');
      if (badge) badge.classList.toggle('vn-visible', _autoMode);
      return _autoMode;
    },
  };

  /* ------------------------------------------------------------------
     History toggle (H key or background click when idle)
     ------------------------------------------------------------------ */
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
      return;
    }
    if (e.code === 'KeyH') {
      $history.classList.toggle('vn-visible');
    }
    if (e.code === 'KeyA') {
      window.VN.toggleAuto();
    }
  });
})();
