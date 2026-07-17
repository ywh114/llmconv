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
  const $dialogue = document.getElementById('vn-dialogue');
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
    session_token: '',
  };

  let _running = false;
  let _loopActive = false;
  let _typingTimer = null;
  let _typingResolve = null;
  let _fullRuns = null;
  let _autoMode = false;
  let _typeGen = 0;
  let _waitResolve = null;
  let _autoWaiter = null;
  let _abortController = null;
  let _transitioning = false;
  let _transitionStartTime = 0;
  let _initialLoad = false;
  let _consecutiveFailures = 0;
  let _lastHereChars = null;
  let _lastSpeakerName = null;
  let _resizeTimer = null;
  const MIN_TRANSITION_MS = 3000;
  const MAX_STEP_FAILURES = 5;
  const MAX_HISTORY = 1000;

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
    // Backgrounds always come from the location dict; there is no bare-name
    // fallback (the old /assets/bg/ path does not exist on the server).
    if (!loc || typeof loc !== 'object' || !loc.background_url) return;
    $bg.style.backgroundImage = `url('${assetUrl('', loc.background_url)}')`;
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

  function isMobileViewport() {
    return window.innerWidth <= 768;
  }

  function getSlots(count) {
    if (count <= 0) return [];
    const mobile = isMobileViewport();

    if (count <= 4) {
      // On mobile, raise sprites into the dedicated upper sprite area so they
      // use vertical space instead of piling up behind the dialogue box.
      if (mobile) {
        if (count === 1) return [{ left: 50, top: 82, scale: 0.78 }];
        if (count === 2) return [
          { left: 25, top: 82, scale: 0.68 },
          { left: 75, top: 82, scale: 0.68 },
        ];
        if (count === 3) {
          // 3 characters fit side-by-side on mobile when scaled down.
          return distribute(3, 86, 82, 0.55);
        }
        // 4 characters: a compact 2x2 grid.
        return [
          { left: 25, top: 86, scale: 0.52 },
          { left: 75, top: 86, scale: 0.52 },
          { left: 25, top: 58, scale: 0.52 },
          { left: 75, top: 58, scale: 0.52 },
        ];
      }
      // Improved desktop layouts for small casts; 4+ keeps the original spread.
      if (count === 1) return [{ left: 50, top: 100, scale: 1.0 }];
      if (count === 2) return [
        { left: 25, top: 100, scale: 0.9 },
        { left: 75, top: 100, scale: 0.9 },
      ];
      if (count === 3) return [
        { left: 28, top: 100, scale: 0.88 },
        { left: 50, top: 100, scale: 0.92 },
        { left: 72, top: 100, scale: 0.88 },
      ];
      return distribute(count, 84, 100, 0.85);
    }

    // 5–8 characters: stacked rows. Higher rows are in front because their
    // sprite artwork extends upward and should overlap the row below.
    if (mobile) {
      const bottom = [
        { left: 18, top: 88, scale: 0.62 },
        { left: 50, top: 88, scale: 0.62 },
        { left: 82, top: 88, scale: 0.62 },
      ];
      const middle = [
        { left: 32, top: 68, scale: 0.56 },
        { left: 68, top: 68, scale: 0.56 },
      ];
      const topRow = [
        { left: 25, top: 52, scale: 0.52 },
        { left: 75, top: 52, scale: 0.52 },
      ];
      if (count === 5) return bottom.concat(middle);
      if (count === 6) return bottom.concat(middle).concat(topRow.slice(0, 1));
      if (count === 7) return bottom.concat(middle).concat(topRow.slice(0, 2));
      return bottom.concat(middle).concat(topRow); // 8
    }

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

  function updateSprites(hereChars, speakerName, forceRecrop) {
    // Remember the latest cast/speaker so the resize handler can re-run the
    // layout without waiting for the next server event.
    _lastHereChars = hereChars;
    _lastSpeakerName = speakerName;
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

    // Fade out sprites that left or switched to sprite='none'.
    Array.from($sprites.children).forEach(el => {
      const char = visibleChars.find(c => c.name === el.dataset.name);
      const spriteName = char ? (char.current_sprite || 'default_neutral') : null;
      if (!char || spriteName === 'none') {
        if (isZoom) {
          clearSpriteAndTimer(el);
          el.remove();
        } else {
          el.classList.add('vn-sprite-hidden');
          setTimeout(() => {
            if (el.parentNode) {
              clearSpriteAndTimer(el);
              el.remove();
            }
          }, 600);
        }
      }
    });

    // Characters with no sprite (e.g. Spectator) should not occupy a slot.
    const displayChars = visibleChars.filter(c => {
      const spriteName = c.current_sprite || 'default_neutral';
      return spriteName !== 'none';
    });

    const slots = isZoom ? [] : getSlots(displayChars.length);

    displayChars.forEach((c, i) => {
      const slot = slots[i];
      const isSpeaking = c.name === speakerName;
      const spriteName = c.current_sprite || 'default_neutral';
      let el = $sprites.querySelector(`[data-name="${c.name}"]`);

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
          wrapper.className = 'vn-sprite-crop vn-zoom vn-sprite-hidden';
          wrapper.dataset.name = c.name;
          wrapper.dataset.sprite = spriteName;
          wrapper._crop = crop || {};
          $sprites.appendChild(wrapper);

          const img = document.createElement('img');
          img.alt = c.name;
          wrapper.appendChild(img);

          img.onload = () => {
            wrapper.classList.remove('vn-sprite-hidden');
            applyZoomFocus(wrapper, img);
          };
          img.onerror = () => {
            clearSpriteAndTimer(wrapper);
            wrapper.remove();
          };
          img.src = url;
          startFocusLoop(wrapper, img);
          el = wrapper;
        } else if (crop && crop.topleft && crop.bottomright) {
          const wrapper = document.createElement('div');
          wrapper.className = 'vn-sprite-crop vn-sprite-hidden';
          wrapper.dataset.name = c.name;
          wrapper.dataset.sprite = spriteName;
          wrapper._crop = crop;
          $sprites.appendChild(wrapper);

          const img = document.createElement('img');
          img.alt = c.name;
          wrapper.appendChild(img);

          const tag = document.createElement('div');
          tag.className = 'vn-nametag';
          tag.textContent = c.display_name || c.name;
          wrapper.appendChild(tag);

          img.onload = () => {
            applyCropStyles(wrapper, img);
            wrapper.classList.remove('vn-sprite-hidden');
          };
          img.onerror = () => {
            clearSpriteAndTimer(wrapper);
            wrapper.remove();
          };
          img.src = url;
          el = wrapper;
        } else {
          const wrapper = document.createElement('div');
          wrapper.className = 'vn-sprite vn-sprite-hidden';
          wrapper.dataset.name = c.name;
          wrapper.dataset.sprite = spriteName;
          $sprites.appendChild(wrapper);

          const img = document.createElement('img');
          img.alt = c.name;
          img.src = url;
          wrapper.appendChild(img);

          const tag = document.createElement('div');
          tag.className = 'vn-nametag';
          tag.textContent = c.display_name || c.name;
          wrapper.appendChild(tag);

          const tempImg = new Image();
          tempImg.onload = () => wrapper.classList.remove('vn-sprite-hidden');
          tempImg.onerror = () => {
            clearSpriteAndTimer(wrapper);
            wrapper.remove();
          };
          tempImg.src = url;
          el = wrapper;
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
      el.classList.remove('vn-sprite-hidden');

      if (forceRecrop) {
        if (!isZoom && el._crop) {
          const img = el.querySelector('img');
          if (img) applyCropStyles(el, img);
        }
        if (isZoom && el._crop) {
          const img = el.querySelector('img');
          if (img) applyZoomFocus(el, img);
        }
      }
    });

    // Bring the speaking sprite to the front in normal layouts.
    if (!isZoom) {
      displayChars.forEach(c => {
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
  let _measurer = null;

  function splitIntoPages(text) {
    const style = getComputedStyle($text);
    const maxHeight = parseFloat(style.maxHeight);
    if (!maxHeight || !text) return [text];

    if (!_measurer) {
      _measurer = document.createElement('div');
      _measurer.style.position = 'absolute';
      _measurer.style.visibility = 'hidden';
      _measurer.style.left = '-9999px';
      _measurer.style.whiteSpace = 'pre-wrap';
      _measurer.style.wordBreak = 'break-word';
      document.body.appendChild(_measurer);
    }
    _measurer.style.width = $text.clientWidth + 'px';
    _measurer.style.font = style.font;
    _measurer.style.lineHeight = style.lineHeight;

    const pages = [];
    let remaining = text;

    while (remaining.length > 0) {
      let low = 1, high = remaining.length, best = 1;
      while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        _measurer.textContent = remaining.slice(0, mid);
        if (_measurer.offsetHeight <= maxHeight) {
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

    return pages.length ? pages : [text];
  }

  /* ------------------------------------------------------------------
     Inline markdown (bold, italic, strikethrough)
     ------------------------------------------------------------------ */
  function parseInlineMarkdown(text) {
    const runs = [];
    const markers = [
      { pattern: '***', classes: ['vn-bold', 'vn-italic'] },
      { pattern: '**', classes: ['vn-bold'] },
      { pattern: '*', classes: ['vn-italic'] },
      { pattern: '_', classes: ['vn-italic'] },
      { pattern: '~~', classes: ['vn-strikethrough'] },
    ];

    function appendPlain(str) {
      const last = runs.length - 1;
      if (last >= 0 && runs[last].classes.length === 0) {
        runs[last].text += str;
      } else {
        runs.push({ text: str, classes: [] });
      }
    }

    let i = 0;
    while (i < text.length) {
      // Escaped marker character: backslash + marker-start prints literally.
      if (text[i] === '\\' && i + 1 < text.length) {
        const next = text[i + 1];
        const isMarkerStart = markers.some(m => m.pattern[0] === next);
        if (isMarkerStart) {
          appendPlain(next);
          i += 2;
          continue;
        }
      }

      let matched = false;
      let prefixMatched = false;
      for (const marker of markers) {
        const len = marker.pattern.length;
        if (text.slice(i, i + len) !== marker.pattern) continue;
        prefixMatched = true;
        const close = text.indexOf(marker.pattern, i + len);
        if (close === -1) continue;
        const content = text.slice(i + len, close);
        if (!content) continue;
        runs.push({ text: content, classes: marker.classes });
        i = close + len;
        matched = true;
        break;
      }
      if (matched) continue;
      if (prefixMatched) {
        // A marker prefix matched but had no valid closing; consume the
        // longest matching prefix as plain text so shorter markers don't
        // misparse the remainder.
        let maxPrefixLen = 0;
        for (const marker of markers) {
          const len = marker.pattern.length;
          if (text.slice(i, i + len) === marker.pattern) {
            maxPrefixLen = Math.max(maxPrefixLen, len);
          }
        }
        appendPlain(text.slice(i, i + maxPrefixLen));
        i += maxPrefixLen;
      } else {
        appendPlain(text[i]);
        i++;
      }
    }
    return runs;
  }

  function appendRunsToText(runs) {
    runs.forEach(run => {
      const span = document.createElement('span');
      if (run.classes.length) span.className = run.classes.join(' ');
      span.textContent = run.text;
      $text.insertBefore(span, $text.lastElementChild);
    });
  }

  /* ------------------------------------------------------------------
     Text typing
     ------------------------------------------------------------------ */
  // Coalesce typed characters back into runs for skipTyping() to render.
  // classes arrays are shared references, so identity comparison is enough.
  function charsToRuns(chars) {
    const runs = [];
    chars.forEach(c => {
      const last = runs[runs.length - 1];
      if (last && last.classes === c.classes) {
        last.text += c.char;
      } else {
        runs.push({ text: c.char, classes: c.classes });
      }
    });
    return runs;
  }

  async function typeText(speaker, text, speakerTitle) {
    const myGen = ++_typeGen;
    $name.textContent = speaker || '';
    let title = speakerTitle || '';
    if (!title && speaker) {
      const char = STATE.pool.find(c => c.name === speaker);
      if (char && char.title) title = char.title;
    }
    $title.textContent = title;

    // Parse styling before paginating: markers are resolved on the full
    // text, and pages are sliced from the styled characters, so a page
    // boundary can never split a marker or leave one unmatched.
    const runs = parseInlineMarkdown(text);
    const allChars = [];
    runs.forEach(run => {
      for (let j = 0; j < run.text.length; j++) {
        allChars.push({ char: run.text[j], classes: run.classes });
      }
    });
    const plainText = allChars.map(c => c.char).join('');
    const pages = splitIntoPages(plainText);

    let offset = 0;
    for (let p = 0; p < pages.length; p++) {
      if (myGen !== _typeGen) {
        return;
      }
      const chars = allChars.slice(offset, offset + pages[p].length);
      offset += pages[p].length;
      _fullRuns = charsToRuns(chars);
      $text.innerHTML = '<span class="vn-cursor"></span>';
      let i = 0;

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
          const item = chars[i];
          const span = document.createElement('span');
          if (item.classes.length) span.className = item.classes.join(' ');
          span.textContent = item.char;
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
    if (_fullRuns) {
      $text.innerHTML = '<span class="vn-cursor"></span>';
      appendRunsToText(_fullRuns);
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
    if (STATE.history.length > MAX_HISTORY) {
      STATE.history.shift();
      if ($historyList.firstChild) $historyList.removeChild($historyList.firstChild);
    }
    const entry = document.createElement('div');
    entry.className = 'vn-history-entry';
    const nameEl = document.createElement('div');
    nameEl.className = 'vn-history-name';
    nameEl.textContent = speaker || 'Narrator';
    const textEl = document.createElement('div');
    textEl.className = 'vn-history-text';
    textEl.textContent = text;
    entry.appendChild(nameEl);
    entry.appendChild(textEl);
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
  function applyEnterExit(enter, exit, spawn) {
    // Merge newly spawned anonymous characters into the pool and here list.
    (spawn || []).forEach(name => {
      if (!STATE.here.find(c => c.name === name)) {
        const char = STATE.pool.find(c => c.name === name);
        if (char) STATE.here.push(char);
      }
    });
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

  function mergeCharacterPool(characters) {
    if (!Array.isArray(characters)) return;
    characters.forEach(c => {
      const idx = STATE.pool.findIndex(existing => existing.name === c.name);
      if (idx === -1) {
        STATE.pool.push(c);
      } else {
        STATE.pool[idx] = c;
      }
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
      if (ev.scene && ev.scene.characters) mergeCharacterPool(ev.scene.characters);
      applyEnterExit(ev.enter, ev.exit, ev.spawn);
      if (ev.sprite_changes) {
        Object.entries(ev.sprite_changes).forEach(([name, sprite]) => {
          const char = STATE.pool.find(c => c.name === name);
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
      await typeText(ev.speaker || null, output, ev.speaker_title);
      return 'wait';
    }

    if (ev.type === 'needs_player_input') {
      if (ev.scene && ev.scene.characters) mergeCharacterPool(ev.scene.characters);
      applyEnterExit(ev.enter, ev.exit, ev.spawn);
      if (ev.sprite_changes) {
        Object.entries(ev.sprite_changes).forEach(([name, sprite]) => {
          const char = STATE.pool.find(c => c.name === name);
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
        if (_autoWaiter === maybeStartAuto) {
          _autoWaiter = null;
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
            e.target.closest('#vn-system') ||
            e.target.closest('#vn-menu-btn') ||
            e.target.closest('#vn-eye-btn') ||
            e.target.closest('#vn-menu')) {
          return;
        }
        if (window.VN.isDialogueCollapsed()) {
          window.VN.restoreDialogue();
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
          if (window.VN.isDialogueCollapsed()) {
            window.VN.restoreDialogue();
            return;
          }
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

      // Event-driven auto mode: toggleAuto() calls the active waiter directly.
      _autoWaiter = maybeStartAuto;
      maybeStartAuto();
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
          _consecutiveFailures++;
          console.error('/step failed', await resp.text());
          if (_consecutiveFailures >= MAX_STEP_FAILURES) {
            $text.innerHTML = '<span style="color:#f55">Connection lost. Refresh to reconnect.</span>';
            _running = false;
            break;
          }
          await sleep(1000);
          continue;
        }
        const ev = await resp.json();
        console.log('Step event:', ev);
        _consecutiveFailures = 0;

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
        _consecutiveFailures++;
        console.error('Game loop error', err);
        if (_consecutiveFailures >= MAX_STEP_FAILURES) {
          $text.innerHTML = '<span style="color:#f55">Connection lost. Refresh to reconnect.</span>';
          _running = false;
          break;
        }
        await sleep(1000);
      }
    }
    _loopActive = false;
  }

  /* ------------------------------------------------------------------
     Network
     ------------------------------------------------------------------ */
  function sessionHeaders() {
    return {
      'Content-Type': 'application/json',
      'X-Session-Token': STATE.session_token,
    };
  }

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
    const data = await resp.json();
    if (data.session_token) STATE.session_token = data.session_token;
    return data;
  }

  async function postInput(text, attempt) {
    const body = { text };
    if (attempt) body.attempt = attempt;
    const resp = await fetch('/input', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('Input failed');
    return resp.json();
  }

  async function postGenerate(suggestion) {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify({ suggestion }),
    });
    if (!resp.ok) throw new Error('Generate failed');
    return resp.json();
  }

  async function postSave(slot) {
    const resp = await fetch('/save', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify({ slot }),
    });
    if (!resp.ok) throw new Error('Save failed');
    return resp.json();
  }

  async function postLoad(slot, storyId) {
    const body = { slot };
    if (storyId) body.story_id = storyId;
    const resp = await fetch('/load', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('Load failed');
    const data = await resp.json();
    if (data.session_token) STATE.session_token = data.session_token;
    return data;
  }

  async function getSaves(storyId) {
    const url = storyId ? `/saves?story_id=${encodeURIComponent(storyId)}` : '/saves';
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('List saves failed');
    return resp.json();
  }

  async function postDelete(slot, storyId) {
    const body = { slot };
    if (storyId) body.story_id = storyId;
    const resp = await fetch('/delete-save', {
      method: 'POST',
      headers: sessionHeaders(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('Delete failed');
    return resp.json();
  }

  async function postDebug(command, args) {
    const resp = await fetch('/debug', {
      method: 'POST',
      headers: sessionHeaders(),
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
    try {
      addToHistory(STATE.player || 'Player', text);
      await typeText(STATE.player || 'Player', text);
      await waitForClick();
      await postInput(text);
      gameLoop();
    } catch (err) {
      console.error('Input failed:', err);
      showChoices([text]);
    }
  }

  async function generateAndSubmit(suggestion) {
    hideChoices();
    try {
      const data = await postGenerate(suggestion);
      const text = data.text || suggestion;
      addToHistory(STATE.player || 'Player', text);
      await typeText(STATE.player || 'Player', text);
      await waitForClick();
      await postInput(text);
      gameLoop();
    } catch (err) {
      console.error('Generate failed:', err);
      showChoices([suggestion]);
    }
  }

  async function submitCustom(text, attempt) {
    hideChoices();
    try {
      addToHistory(STATE.player || 'Player', text);
      await typeText(STATE.player || 'Player', text);
      await waitForClick();
      await postInput(text, attempt);
      gameLoop();
    } catch (err) {
      console.error('Custom input failed:', err);
      showChoices([text]);
    }
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
      const nameEl = document.createElement('div');
      nameEl.className = 'vn-history-name';
      nameEl.textContent = entry.speaker || 'Narrator';
      const textEl = document.createElement('div');
      textEl.className = 'vn-history-text';
      textEl.textContent = entry.text;
      div.appendChild(nameEl);
      div.appendChild(textEl);
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
      if (_autoWaiter) {
        _autoWaiter();
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
    async load(slot, storyId) {
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
        data = await postLoad(slot, storyId);
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
    async listSaves(storyId) {
      return getSaves(storyId);
    },
    async delete(slot, storyId) {
      showLoading('Deleting…');
      try {
        return await postDelete(slot, storyId);
      } finally {
        hideLoading();
      }
    },
    async debug(command, args) {
      return postDebug(command, args || []);
    },
    isDialogueCollapsed() {
      return $dialogue && $dialogue.classList.contains('vn-collapsed');
    },
    collapseDialogue() {
      if ($dialogue) $dialogue.classList.add('vn-collapsed');
    },
    restoreDialogue() {
      if ($dialogue) $dialogue.classList.remove('vn-collapsed');
    },
    setVisibilityMode(mode) {
      document.body.classList.remove('vn-hide-nametags');
      if (mode === 1) document.body.classList.add('vn-hide-nametags');
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
        e.stopImmediatePropagation();
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

  /* ------------------------------------------------------------------
     Click / tap to skip typing (mirrors the Space key handler above).
     waitForClick() only listens after typing finishes, so without this
     a click during the typewriter effect did nothing.
     ------------------------------------------------------------------ */
  document.addEventListener('click', (e) => {
    if (!isTyping()) return;
    if (e.target.closest('.vn-choice') ||
        e.target.closest('.vn-title-btn') ||
        e.target.closest('.vn-control-btn') ||
        e.target.closest('#vn-controls') ||
        e.target.closest('#vn-settings') ||
        e.target.closest('#vn-history') ||
        e.target.closest('#vn-keybinds') ||
        e.target.closest('#vn-saveload') ||
        e.target.closest('#vn-debug') ||
        e.target.closest('#vn-system') ||
        e.target.closest('#vn-menu-btn') ||
        e.target.closest('#vn-eye-btn') ||
        e.target.closest('#vn-collapse-btn') ||
        e.target.closest('#vn-menu') ||
        e.target.closest('input') ||
        e.target.closest('textarea')) {
      return;
    }
    // Collapsed: restoreIfCollapsed() restores instead of skipping.
    if (window.VN.isDialogueCollapsed()) return;
    skipTyping();
  });

  /* ------------------------------------------------------------------
     Mobile tap-to-skip-typing on the dialogue box
     ------------------------------------------------------------------ */
  if ($dialogue) {
    $dialogue.addEventListener('touchstart', (e) => {
      if (e.target.closest('#vn-controls') ||
          e.target.closest('#vn-choices') ||
          e.target.closest('input') ||
          e.target.closest('textarea')) {
        return;
      }
      if (window.VN.isDialogueCollapsed()) {
        e.preventDefault();
        window.VN.restoreDialogue();
        return;
      }
      if (isTyping()) {
        e.preventDefault();
        skipTyping();
      }
    }, { passive: false });
  }

  // Global tap/click restore for collapsed dialogue.
  function restoreIfCollapsed(e) {
    if (!window.VN.isDialogueCollapsed()) return;
    if (e.target.closest('#vn-menu') ||
        e.target.closest('#vn-menu-btn') ||
        e.target.closest('#vn-eye-btn')) {
      return;
    }
    if (e.type === 'touchstart') e.preventDefault();
    window.VN.restoreDialogue();
  }
  document.addEventListener('click', restoreIfCollapsed);
  document.addEventListener('touchstart', restoreIfCollapsed, { passive: false });

  // Recalculate sprite layout when the viewport changes.
  window.addEventListener('resize', () => {
    if (_resizeTimer) clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      _resizeTimer = null;
      if (_lastHereChars) updateSprites(_lastHereChars, _lastSpeakerName, true);
    }, 150);
  });
})();
