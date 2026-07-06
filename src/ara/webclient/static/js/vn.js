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
  const $title = document.getElementById('vn-speaker-title');

  function clearSpeaker() {
    $name.textContent = '';
    $title.textContent = '';
  }
  const $text = document.getElementById('vn-text');
  const $choices = document.getElementById('vn-choices');
  const $history = document.getElementById('vn-history');
  const $historyList = document.getElementById('vn-history-list');
  const $transition = document.getElementById('vn-transition');
  const $transitionNext = document.getElementById('vn-transition-next');

  /* ------------------------------------------------------------------
     State
     ------------------------------------------------------------------ */
  const STATE = {
    pool: [],
    here: [],
    narrator: null,
    player: null,
    story_id: '',
    history: [],
    textSpeed: 30,
    autoDelay: 2500,
  };

  let _running = false;
  let _loopActive = false;
  let _typingTimer = null;
  let _typingResolve = null;
  let _fullText = '';
  let _pendingClick = null;
  let _autoMode = false;
  let _abortTyping = false;
  let _typeGen = 0;
  let _waitResolve = null;
  let _abortController = null;
  let _transitioning = false;
  let _transitionStartTime = 0;
  let _initialLoad = false;
  const MIN_TRANSITION_MS = 3000;

  /* ------------------------------------------------------------------
     Helpers
     ------------------------------------------------------------------ */
  function assetUrl(type, name) {
    let url = '/assets/';
    if (type) url += type + '/';
    if (type === 'cc' && STATE.story_id) url += STATE.story_id + '/';
    url += name;
    return url;
  }

  function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  /* ------------------------------------------------------------------
     Background
     ------------------------------------------------------------------ */
  function setBackground(loc) {
    if (!loc) return;
    let url;
    if (typeof loc === 'object' && loc.background_url) {
      url = assetUrl('', loc.background_url);
    } else if (typeof loc === 'string') {
      url = assetUrl('bg', loc + '.png');
    } else if (typeof loc === 'object' && loc.name) {
      url = assetUrl('bg', loc.name + '.png');
    }
    if (url) $bg.style.backgroundImage = `url('${url}')`;
  }

  /* ------------------------------------------------------------------
     Sprites
     ------------------------------------------------------------------ */
  function clearSpriteAndTimer(el) {
    if (el && el._focusTimer) {
      clearTimeout(el._focusTimer);
      el._focusTimer = null;
    }
  }

  function clearSprites() {
    Array.from($sprites.children).forEach(clearSpriteAndTimer);
    $sprites.innerHTML = '';
    $sprites.classList.remove('vn-zoom-layout');
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

    // Center the crop region on the explicit center point from meta.toml if
    // provided, otherwise on the geometric midpoint of the crop rectangle.
    const hasCenter = crop.center && crop.center[0] != null && crop.center[1] != null;
    const centerX = hasCenter ? crop.center[0] : x1 + cropW / 2;
    const centerY = hasCenter ? crop.center[1] : y1 + cropH / 2;
    const offsetX = -centerX * scale;
    const offsetY = (fullH - centerY) * scale - renderH / 2;

    const lift = wrapper.classList.contains('vn-speaking') ? -8 : 0;
    img.style.transform = `translate(${offsetX}px, ${offsetY + lift}px)`;
  }

  function applyZoomFocus(wrapper, img) {
    if (!img.naturalWidth) return;

    const crop = wrapper._crop || {};
    const focusList = crop.focus;
    let focus;
    if (focusList && focusList.length) {
      const idx = Math.floor(Math.random() * focusList.length);
      focus = focusList[idx];
    }
    if (!focus || focus.length < 3) {
      const r = Math.min(img.naturalWidth, img.naturalHeight) / 2;
      focus = [img.naturalWidth / 2, img.naturalHeight / 2, r];
    }

    const [cx, cy, r] = focus;
    if (!r || r <= 0) return;

    const size = wrapper.clientWidth || wrapper.offsetWidth || 120;
    const scale = size / (2 * r);
    const tx = size / 2 - cx * scale;
    const ty = size / 2 - cy * scale;

    img.style.width = `${img.naturalWidth * scale}px`;
    img.style.height = `${img.naturalHeight * scale}px`;
    img.style.transform = `translate(${tx}px, ${ty}px)`;
  }

  function startFocusLoop(wrapper, img) {
    const crop = wrapper._crop || {};
    const focusList = crop.focus;
    if (!focusList || focusList.length <= 1) return;

    function step() {
      if (!wrapper.parentNode) return;
      applyZoomFocus(wrapper, img);
      wrapper._focusTimer = setTimeout(step, 5000 + Math.random() * 5000);
    }

    wrapper._focusTimer = setTimeout(step, 5000 + Math.random() * 5000);
  }

  function distribute(n, width, top, scale) {
    const slots = [];
    if (n <= 0) return slots;
    if (n === 1) {
      slots.push({ left: 50, top, scale });
      return slots;
    }

    const margin = (100 - width) / 2;
    const step = width / (n - 1);
    for (let i = 0; i < n; i++) {
      slots.push({ left: margin + i * step, top, scale });
    }
    return slots;
  }

  function getSlots(count) {
    if (count <= 0) return [];
    if (count <= 4) {
      return distribute(count, 84, 100, 0.85);
    }

    // 5–8 characters: stacked rows. Higher rows are in front because their
    // sprite artwork extends upward and should overlap the row below.
    const bottom = [
      { left: 15, top: 100, scale: 0.85 },
      { left: 50, top: 100, scale: 0.85 },
      { left: 85, top: 100, scale: 0.85 },
    ];
    const middle = [
      { left: 33, top: 70, scale: 0.68 },
      { left: 67, top: 70, scale: 0.68 },
    ];
    // Keep the top row fully inside the viewport. At top:40 the effective
    // sprite area extended above the screen, so we lower the row and scale
    // down slightly.
    const top = [
      { left: 15, top: 55, scale: 0.65 },
      { left: 50, top: 55, scale: 0.65 },
      { left: 85, top: 55, scale: 0.65 },
    ];

    if (count === 5) return bottom.concat(middle);
    if (count === 6) return bottom.concat(middle).concat(top.slice(0, 1));
    if (count === 7) return bottom.concat(middle).concat(top.slice(0, 2));
    return bottom.concat(middle).concat(top); // 8
  }

  function updateSprites(hereChars, speakerName) {
    const playerName = STATE.player;
    const visibleChars = hereChars.filter(c => {
      if (c.name === STATE.narrator) return false;
      if (c.hidden) {
        return playerName && (playerName === c.name || (c.visible_to || []).includes(playerName));
      }
      return true;
    });
    const isZoom = visibleChars.length > 8;
    const wasZoom = $sprites.classList.contains('vn-zoom-layout');

    // Switching between normal and zoom layouts requires a full rebuild.
    if (isZoom !== wasZoom) {
      Array.from($sprites.children).forEach(clearSpriteAndTimer);
      $sprites.innerHTML = '';
      $sprites.classList.toggle('vn-zoom-layout', isZoom);
    }

    // Fade out sprites that left.
    Array.from($sprites.children).forEach(el => {
      if (!visibleChars.find(c => c.name === el.dataset.name)) {
        if (isZoom) {
          clearSpriteAndTimer(el);
          el.remove();
        } else {
          el.classList.add('vn-hidden');
          setTimeout(() => {
            if (el.parentNode) {
              clearSpriteAndTimer(el);
              el.remove();
            }
          }, 600);
        }
      }
    });

    const slots = isZoom ? [] : getSlots(visibleChars.length);

    visibleChars.forEach((c, i) => {
      const slot = slots[i];
      const isSpeaking = c.name === speakerName;
      const spriteName = c.current_sprite || 'default_neutral';
      let el = $sprites.querySelector(`[data-name="${c.name}"]`);

      if (spriteName === 'none') {
        if (el) {
          clearSpriteAndTimer(el);
          el.remove();
        }
        return;
      }

      const needsRebuild = el && (
        el.dataset.sprite !== spriteName ||
        (isZoom && !el.classList.contains('vn-zoom')) ||
        (!isZoom && el.classList.contains('vn-zoom'))
      );
      if (needsRebuild) {
        clearSpriteAndTimer(el);
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
        const crop = c.crops && c.crops[spriteName];

        if (isZoom) {
          const wrapper = document.createElement('div');
          wrapper.className = 'vn-sprite-crop vn-zoom vn-hidden';
          wrapper.dataset.name = c.name;
          wrapper.dataset.sprite = spriteName;
          wrapper._crop = crop || {};
          $sprites.appendChild(wrapper);

          const img = document.createElement('img');
          img.alt = c.name;
          wrapper.appendChild(img);

          img.onload = () => {
            wrapper.classList.remove('vn-hidden');
            applyZoomFocus(wrapper, img);
          };
          img.onerror = () => {
            if (!isAnon) img.src = assetUrl('cc', 'Cheshire/default_neutral.png');
            wrapper.classList.remove('vn-hidden');
          };
          img.src = url;
          startFocusLoop(wrapper, img);
          el = wrapper;
        } else if (crop && crop.topleft && crop.bottomright) {
          const wrapper = document.createElement('div');
          wrapper.className = 'vn-sprite-crop vn-hidden';
          wrapper.dataset.name = c.name;
          wrapper.dataset.sprite = spriteName;
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
            if (!isAnon) img.src = assetUrl('cc', 'Cheshire/default_neutral.png');
            wrapper.classList.remove('vn-hidden');
          };
          img.src = url;
          el = wrapper;
        } else {
          el = document.createElement('img');
          el.className = 'vn-sprite vn-hidden';
          el.dataset.name = c.name;
          el.dataset.sprite = spriteName;
          el.alt = c.name;
          $sprites.appendChild(el);

          el.src = url;
          const tempImg = new Image();
          tempImg.onload = () => el.classList.remove('vn-hidden');
          tempImg.onerror = () => {
            if (!isAnon) el.src = assetUrl('cc', 'Cheshire/default_neutral.png');
            el.classList.remove('vn-hidden');
          };
          tempImg.src = url;
        }
      }

      if (!isZoom && slot) {
        el.style.setProperty('--slot-left', slot.left);
        el.style.setProperty('--slot-top', slot.top);
        el.style.setProperty('--slot-scale', slot.scale);
        // Sprites higher on the screen are in front; their artwork extends
        // upward and should overlap the characters standing behind them.
        const baseZ = Math.round(150 - slot.top);
        // The speaker must always be visible on top, even when standing in
        // the back row.
        el.style.zIndex = isSpeaking ? '999' : String(baseZ);
      }
      el.classList.toggle('vn-speaking', isSpeaking);
      el.classList.remove('vn-hidden');

      if (!isZoom && el._crop) {
        const img = el.querySelector('img');
        if (img) applyCropStyles(el, img);
      }
      if (isZoom && el._crop) {
        const img = el.querySelector('img');
        if (img) applyZoomFocus(el, img);
      }
    });

    // Bring the speaking sprite to the front in normal layouts.
    if (!isZoom) {
      visibleChars.forEach(c => {
        if (c.name === speakerName) {
          const el = $sprites.querySelector(`[data-name="${c.name}"]`);
          if (el) $sprites.appendChild(el);
        }
      });
    }
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
    const myGen = ++_typeGen;
    const pages = splitIntoPages(text);
    $name.textContent = speaker || '';
    const char = speaker ? STATE.pool.find(c => c.name === speaker) : null;
    $title.textContent = char && char.title ? char.title : '';

    for (let p = 0; p < pages.length; p++) {
      if (myGen !== _typeGen) {
        return;
      }
      _fullText = pages[p];
      $text.innerHTML = '<span class="vn-cursor"></span>';
      let i = 0;
      const chars = pages[p].split('');

      await new Promise(resolve => {
        _typingResolve = resolve;
        function tick() {
          if (myGen !== _typeGen) {
            _typingTimer = null;
            _typingResolve = null;
            resolve();
            return;
          }
          if (i >= chars.length) {
            _typingTimer = null;
            _typingResolve = null;
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
    if (_typingResolve) {
      _typingResolve();
      _typingResolve = null;
    }
  }

  function isTyping() {
    return _typingTimer !== null;
  }

  /* ------------------------------------------------------------------
     History
     ------------------------------------------------------------------ */
  function addToHistory(speaker, text) {
    const last = STATE.history[STATE.history.length - 1];
    if (last && last.speaker === speaker && last.text === text) {
      return;
    }
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

    const attemptArea = document.createElement('textarea');
    attemptArea.className = 'vn-attempt-input';
    attemptArea.placeholder = 'Attempt action (optional)...';
    attemptArea.rows = 2;
    $choices.appendChild(attemptArea);

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
        if (text) submitCustom(text, attemptArea.value.trim());
      }
    });
    const send = document.createElement('button');
    send.className = 'vn-custom-send';
    send.textContent = 'Send';
    send.onclick = () => {
      const text = input.value.trim();
      if (text) submitCustom(text, attemptArea.value.trim());
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

  function showTransition(nextScene, nextSceneName, label) {
    console.log('[VN] showTransition called:', nextScene, nextSceneName);
    const labelEl = $transition ? $transition.querySelector('.vn-transition-label') : null;
    if (labelEl) labelEl.textContent = label || 'Scene Finished';
    if ($transitionNext) $transitionNext.textContent = nextSceneName || nextScene || '';
    if ($transition) {
      $transition.style.display = 'flex';
    }
    _transitioning = true;
    _transitionStartTime = Date.now();
  }

  async function hideTransition(skipMinDelay) {
    const elapsed = Date.now() - _transitionStartTime;
    if (!skipMinDelay && elapsed < MIN_TRANSITION_MS) {
      await sleep(MIN_TRANSITION_MS - elapsed);
    }
    if ($transition) $transition.style.display = 'none';
    _transitioning = false;
  }

  async function processEvent(ev) {
    if (ev.type === 'scene_loaded') {
      await hideTransition(_initialLoad);
      _initialLoad = false;
      STATE.narrator = ev.narrator || null;
      STATE.player = ev.player || null;
      STATE.story_id = ev.asset_story_name || STATE.story_id;
      STATE.pool = ev.characters || [];
      const starting = new Set(ev.starting_characters || []);
      STATE.here = (ev.characters || []).filter(c => starting.has(c.name));
      setBackground(ev.location);
      clearSprites();
      updateSprites(STATE.here, null);
      clearSpeaker();
      $text.innerHTML = '';

      if (STATE.opening_text) {
        const text = STATE.opening_text;
        STATE.opening_text = '';
        await typeText(null, text);
        return 'wait';
      }

      return 'continue';
    }

    if (ev.type === 'turn' || ev.type === 'finalize_turn') {
      applyEnterExit(ev.enter, ev.exit);
      if (ev.sprite_changes) {
        Object.entries(ev.sprite_changes).forEach(([name, sprite]) => {
          const char = STATE.here.find(c => c.name === name);
          if (char) char.current_sprite = sprite;
        });
      }
      if (ev.system_changes && window.applySystemChanges) {
        window.applySystemChanges(ev.system_changes);
      }
      if (ev.location) setBackground(ev.location);
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
      if (ev.system_changes && window.applySystemChanges) {
        window.applySystemChanges(ev.system_changes);
      }
      if (ev.location) setBackground(ev.location);
      updateSprites(STATE.here, ev.speaker || null);
      showChoices(ev.suggestions || []);
      return 'input';
    }

    if (ev.type === 'transition' && ev.phase === 'ended') {
      clearSpeaker();
      $text.textContent = '— Scene End —';
      hideChoices();
      if (ev.loading_background) {
        setBackground(ev.loading_background);
      }
      showTransition(ev.next_scene, ev.next_scene_name);
      return 'continue';
    }

    if (ev.type === 'scene_ended') {
      clearSpeaker();
      $text.textContent = '— Scene End —';
      hideChoices();
      if (ev.loading_background) {
        setBackground(ev.loading_background);
      }
      showTransition(ev.next_scene, ev.next_scene_name);
      return 'continue';
    }

    if (ev.type === 'story_complete') {
      clearSpeaker();
      $text.textContent = '— The End —';
      hideChoices();
      await hideTransition();
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
      let pollInterval = null;
      let finished = false;
      _waitResolve = resolve;

      function finish() {
        if (finished) return;
        finished = true;
        _waitResolve = null;
        if (autoTimer) {
          clearTimeout(autoTimer);
          autoTimer = null;
        }
        if (pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
        cleanup();
        resolve();
      }

      function onClick(e) {
        if (e.target.closest('.vn-choice') ||
            e.target.closest('.vn-title-btn') ||
            e.target.closest('.vn-control-btn') ||
            e.target.closest('#vn-controls') ||
            e.target.closest('#vn-settings') ||
            e.target.closest('#vn-history') ||
            e.target.closest('#vn-keybinds') ||
            e.target.closest('#vn-saveload') ||
            e.target.closest('#vn-debug') ||
            e.target.closest('#vn-system')) {
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

      function maybeStartAuto() {
        if (finished) return;
        if (_autoMode && !autoTimer) {
          autoTimer = setTimeout(finish, STATE.autoDelay);
        } else if (!_autoMode && autoTimer) {
          clearTimeout(autoTimer);
          autoTimer = null;
        }
      }

      maybeStartAuto();
      pollInterval = setInterval(maybeStartAuto, 200);
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
        if (!_transitioning) {
          clearSpeaker();
          $text.innerHTML = '<span style="color:var(--vn-gold)">…</span>';
        }

        _abortController = new AbortController();
        const resp = await fetch('/step', {
          method: 'POST',
          signal: _abortController.signal,
        });
        _abortController = null;
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
        if (err.name === 'AbortError') {
          // Fetch was cancelled by load/start/reset – exit cleanly.
          break;
        }
        console.error('Game loop error', err);
        await sleep(1000);
      }
    }
    _loopActive = false;
  }

  /* ------------------------------------------------------------------
     Network
     ------------------------------------------------------------------ */
  async function postStart(sceneId, storyId) {
    const params = {};
    if (sceneId) params.scene_id = sceneId;
    if (storyId) params.story_id = storyId;
    const resp = await fetch('/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!resp.ok) throw new Error('Start failed');
    return resp.json();
  }

  async function postInput(text, attempt) {
    const body = { text };
    if (attempt) body.attempt = attempt;
    const resp = await fetch('/input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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

  async function postSave(slot) {
    const resp = await fetch('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot }),
    });
    if (!resp.ok) throw new Error('Save failed');
    return resp.json();
  }

  async function postLoad(slot) {
    const resp = await fetch('/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot }),
    });
    if (!resp.ok) throw new Error('Load failed');
    return resp.json();
  }

  async function getSaves() {
    const resp = await fetch('/saves');
    if (!resp.ok) throw new Error('List saves failed');
    return resp.json();
  }

  async function postDelete(slot) {
    const resp = await fetch('/delete-save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot }),
    });
    if (!resp.ok) throw new Error('Delete failed');
    return resp.json();
  }

  async function postDebug(command, args) {
    const resp = await fetch('/debug', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, args }),
    });
    if (!resp.ok) throw new Error('Debug failed');
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

  async function submitCustom(text, attempt) {
    hideChoices();
    addToHistory('Player', text);
    await typeText('Player', text);
    await waitForClick();
    await postInput(text, attempt);
    gameLoop();
  }

  /* ------------------------------------------------------------------
     Visual state init (used by start / load / reset)
     ------------------------------------------------------------------ */
  async function initVisualState(data) {
    _typeGen++;
    skipTyping();
    if (_waitResolve) _waitResolve();
    hideChoices();
    if (!_initialLoad) {
      hideTransition();
    }

    const scene = data.scene || {};
    const engine = data.engine || {};
    const hereChars = data.here || [];
    const location = data.location || {};

    STATE.narrator = scene.narrator || null;
    STATE.player = scene.player || null;
    STATE.story_id = scene.asset_story_name || '';
    STATE.pool = scene.characters || [];
    STATE.here = hereChars;

    setBackground(location.background_url ? location : (location.name || scene.starting_location));
    clearSprites();
    updateSprites(STATE.here, data.current_speaker || null);

    clearSpeaker();
    $text.innerHTML = '';

    // History
    STATE.history = [];
    $historyList.innerHTML = '';
    const history = data.history || [];
    history.forEach(entry => {
      STATE.history.push(entry);
      const div = document.createElement('div');
      div.className = 'vn-history-entry';
      div.innerHTML = `
        <div class="vn-history-name">${entry.speaker || 'Narrator'}</div>
        <div class="vn-history-text">${entry.text}</div>
      `;
      $historyList.appendChild(div);
    });

    // Leave the text box empty; gameLoop will drive typing from the queue.
    clearSpeaker();
    $text.innerHTML = '';
  }

  /* ------------------------------------------------------------------
     Loading helpers
     ------------------------------------------------------------------ */
  function showLoading(text) {
    const el = document.getElementById('vn-loading');
    const txt = el?.querySelector('.vn-loading-text');
    if (txt) txt.textContent = text || 'Loading…';
    if (el) el.classList.add('vn-visible');
  }
  function hideLoading() {
    const el = document.getElementById('vn-loading');
    if (el) el.classList.remove('vn-visible');
  }

  /* ------------------------------------------------------------------
     Public API
     ------------------------------------------------------------------ */
  window.VN = {
    async start(sceneId, storyId) {
      // Stop any running game loop before starting fresh
      _running = false;
      if (_abortController) {
        _abortController.abort();
        _abortController = null;
      }
      _typeGen++;
      skipTyping();
      if (_waitResolve) _waitResolve();
      while (_loopActive) {
        if (_waitResolve) _waitResolve();
        await sleep(50);
      }
      _running = true;
      _initialLoad = true;
      showTransition(null, '…', 'Loading');
      const data = await postStart(sceneId, storyId);
      if (data) {
        STATE.opening_text = data.opening_text || '';
        await initVisualState(data);
        gameLoop();
      }
    },
    setTextSpeed(v) {
      STATE.textSpeed = parseInt(v, 10);
    },
    setAutoDelay(v) {
      STATE.autoDelay = parseInt(v, 10);
    },
    toggleHistory() {
      $history.classList.toggle('vn-visible');
    },
    toggleAuto() {
      _autoMode = !_autoMode;
      const indicator = document.getElementById('vn-auto-indicator');
      if (indicator) {
        indicator.style.display = _autoMode ? 'block' : 'none';
      }
      return _autoMode;
    },
    showLoading,
    hideLoading,
    async save(slot) {
      showLoading('Saving…');
      try {
        return await postSave(slot);
      } finally {
        hideLoading();
      }
    },
    async load(slot) {
      showLoading('Loading…');
      // Stop any running game loop before mutating state
      _running = false;
      if (_abortController) {
        _abortController.abort();
        _abortController = null;
      }
      _typeGen++;
      skipTyping();
      if (_waitResolve) _waitResolve();
      // Poll until the old loop exits; keep force-resolving any new
      // waitForClick that starts while we are waiting.
      while (_loopActive) {
        if (_waitResolve) _waitResolve();
        await sleep(50);
      }
      let data;
      try {
        data = await postLoad(slot);
        _running = true;
      } finally {
        hideLoading();
      }
      if (data) {
        await initVisualState(data);
        gameLoop();
      }
      return data;
    },
    async listSaves() {
      return getSaves();
    },
    async delete(slot) {
      showLoading('Deleting…');
      try {
        return await postDelete(slot);
      } finally {
        hideLoading();
      }
    },
    async debug(command, args) {
      return postDebug(command, args || []);
    },
  };

  /* ------------------------------------------------------------------
     History toggle (H key or background click when idle)
     ------------------------------------------------------------------ */
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
      return;
    }
    if (e.code === 'Space') {
      if (isTyping()) {
        e.preventDefault();
        skipTyping();
      }
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
