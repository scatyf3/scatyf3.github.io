'use strict';
// 只读静态构建(GitHub Pages)的适配层 —— 本地跑 server.py 时这个文件不会被加载。
// 和 dashboard/static-shim.js 同一个套路: fetch 改道到烤好的 .json, 写请求就地拒绝。
//
// 这里还多挡一层: 线上默认**没有** api/apps.json(见 export_static.py 的
// EXPORT_APPLICATIONS), 所以拿不到就退成空列表 —— 公开站只展示方法论, 不展示你投了谁。
(function () {
  const BASE = new URL('.', document.baseURI);
  const realFetch = window.fetch.bind(window);
  const json = (obj, status) =>
    new Response(JSON.stringify(obj), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json' },
    });

  document.documentElement.classList.add('ro');

  window.fetch = function (input, opt) {
    const raw = typeof input === 'string' ? input : input.url;
    const u = new URL(raw, document.baseURI);
    const m = /\/api\/(.+)$/.exec(u.pathname);
    if (!m) return realFetch(input, opt);

    const method = ((opt && opt.method) || 'GET').toUpperCase();
    if (method !== 'GET') return Promise.resolve(json({ ok: false, readonly: true }, 403));

    return realFetch(new URL('api/' + m[1] + '.json', BASE))
      .then((r) => (r.ok ? r : json({ apps: [], events: [] })))
      .catch(() => json({ apps: [], events: [] }));
  };
})();
