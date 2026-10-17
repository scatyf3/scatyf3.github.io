/* Site enhancements: theme toggle, scroll reveal, progress bar, back-to-top. */
(function () {
  'use strict';

  var root = document.documentElement;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Theme toggle ---------- */

  function storedTheme() {
    try { return localStorage.getItem('theme'); } catch (e) { return null; }
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', theme === 'dark' ? '#16181d' : '#ffffff');
  }

  applyTheme(root.getAttribute('data-theme') || 'light');

  var toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
    });
  }

  /* follow the OS setting until the visitor picks a theme explicitly */
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function (e) {
      if (!storedTheme()) applyTheme(e.matches ? 'dark' : 'light');
    };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }

  /* ---------- Language toggle ---------- */

  function applyLang(lang) {
    root.setAttribute('data-lang', lang);
    root.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
    /* nav labels change width; let the greedy nav re-measure */
    window.dispatchEvent(new Event('resize'));
  }

  var langToggle = document.querySelector('.lang-toggle');
  if (langToggle) {
    langToggle.addEventListener('click', function () {
      var next = root.getAttribute('data-lang') === 'zh' ? 'en' : 'zh';
      applyLang(next);
      try { localStorage.setItem('lang', next); } catch (e) { /* ignore */ }
    });
  }

  /* ---------- Readable anchors for 中文 headings ---------- */

  /* kramdown drops CJK characters from auto ids, leaving #section, #section-1, ...;
     rebuild those from the heading text so shared section links are readable */
  var usedIds = {};
  Array.prototype.forEach.call(document.querySelectorAll('[id]'), function (el) { usedIds[el.id] = true; });
  Array.prototype.forEach.call(
    document.querySelectorAll('.page__content h2[id], .page__content h3[id], .page__content h4[id]'),
    function (h) {
      if (!/^section(-\d+)?$/.test(h.id)) return;
      var base = h.textContent.trim().toLowerCase()
        .replace(/[^\w\u3400-\u9fff\s-]/g, '')
        .replace(/\s+/g, '-');
      if (!base) return;
      var id = base;
      for (var n = 1; usedIds[id]; n++) id = base + '-' + n;
      delete usedIds[h.id];
      usedIds[id] = true;
      h.id = id;
    }
  );
  if (location.hash) {
    var hashTarget = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    /* a section link from the other language: show that language for this visit */
    var hashLang = hashTarget && hashTarget.closest('.lang-en, .lang-zh');
    if (hashLang && hashTarget.offsetParent === null) {
      applyLang(hashLang.classList.contains('lang-zh') ? 'zh' : 'en');
    }
    if (hashTarget) hashTarget.scrollIntoView();
  }

  /* ---------- Post table of contents (left sidebar) ---------- */

  var toc = document.querySelector('.post-toc');
  var tocHeadings = toc ? document.querySelectorAll('.page__content h2[id], .page__content h3[id], .page__content h4[id]') : [];

  if (toc && tocHeadings.length) {
    var tocList = toc.querySelector('.post-toc__list');
    var tocLinks = [];
    var topLevel = Math.min.apply(null, Array.prototype.map.call(tocHeadings, function (h) {
      return +h.tagName.charAt(1);
    }));

    Array.prototype.forEach.call(tocHeadings, function (h) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      li.className = 'post-toc__item post-toc__item--l' + (+h.tagName.charAt(1) - topLevel);
      /* headings inside a .lang-en / .lang-zh block only show in that language */
      var langBlock = h.closest('.lang-en, .lang-zh');
      if (langBlock) li.classList.add(langBlock.classList.contains('lang-zh') ? 'lang-zh' : 'lang-en');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      tocList.appendChild(li);
      tocLinks.push(a);
    });
    toc.hidden = false;

    /* highlight the section currently being read */
    var tocTicking = false;
    var updateToc = function () {
      var offset = 120;
      var current = -1;
      Array.prototype.forEach.call(tocHeadings, function (h, i) {
        if (h.offsetParent !== null && h.getBoundingClientRect().top - offset <= 0) current = i;
      });
      tocLinks.forEach(function (a, i) { a.classList.toggle('is-active', i === current); });
      tocTicking = false;
    };
    window.addEventListener('scroll', function () {
      if (!tocTicking) {
        tocTicking = true;
        window.requestAnimationFrame(updateToc);
      }
    }, { passive: true });
    updateToc();
  }

  /* ---------- Scroll reveal ---------- */

  if (!reduceMotion && 'IntersectionObserver' in window) {
    var targets = document.querySelectorAll(
      '.archive .list__item, .page__content > h2, .page__content > ul > li, .page__content > p'
    );
    var observer = new IntersectionObserver(function (entries) {
      var delay = 0;
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.style.transitionDelay = delay + 'ms';
        delay += 60;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '0px 0px -40px 0px' });

    Array.prototype.forEach.call(targets, function (el) {
      el.classList.add('reveal');
      observer.observe(el);
    });
  }

  /* ---------- Reading progress, sticky masthead, back-to-top ---------- */

  var progress = document.querySelector('.reading-progress');
  var masthead = document.querySelector('.masthead');
  var backToTop = document.querySelector('.back-to-top');
  var ticking = false;

  function onScroll() {
    var y = window.pageYOffset || root.scrollTop;
    var max = root.scrollHeight - window.innerHeight;

    if (progress) {
      progress.style.display = max > 200 ? '' : 'none';
      progress.style.transform = 'scaleX(' + (max > 0 ? Math.min(y / max, 1) : 0) + ')';
    }
    if (masthead) masthead.classList.toggle('is-scrolled', y > 40);
    if (backToTop) backToTop.classList.toggle('is-visible', y > window.innerHeight);
    ticking = false;
  }

  window.addEventListener('scroll', function () {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(onScroll);
    }
  }, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();

  if (backToTop) {
    backToTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
    });
  }
})();
