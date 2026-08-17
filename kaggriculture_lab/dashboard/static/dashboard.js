(() => {
  const container = document.createElement('template');

  // Adapted from https://github.com/flying-swallow/alpine-sse. Only the
  // fragment patcher is retained; this dashboard has no Alpine dependency.
  const mergeFragment = ({selector, mergeMode = 'outer', fragments: html}) => {
    container.innerHTML = html.trim();
    for (const fragment of [...container.content.children]) {
      const targets = [...document.querySelectorAll(selector || `#${fragment.id}`)];
      for (const target of targets) {
        const replacement = fragment.cloneNode(true);
        if (mergeMode === 'inner') target.innerHTML = replacement.outerHTML;
        else if (mergeMode === 'outer') target.replaceWith(replacement);
        else if (mergeMode === 'prepend') target.prepend(replacement);
        else if (mergeMode === 'append') target.append(replacement);
        else if (mergeMode === 'before') target.before(replacement);
        else if (mergeMode === 'after') target.after(replacement);
        else if (mergeMode === 'upsertAttributes') {
          for (const name of replacement.getAttributeNames()) {
            target.setAttribute(name, replacement.getAttribute(name));
          }
        }
      }
    }
  };

  let updates;
  const connect = path => {
    if (updates) updates.close();
    const url = new URL('/events/dashboard', location.origin);
    url.searchParams.set('path', path);
    updates = new EventSource(url);
    updates.addEventListener('dashboard.fragment', event => mergeFragment(JSON.parse(event.data)));
  };

  document.addEventListener('submit', event => {
    const form = event.target.closest('form.filters');
    if (!form) return;
    event.preventDefault();
    const target = new URL(form.action || location.href, location.href);
    target.search = new URLSearchParams(new FormData(form)).toString();
    history.pushState({}, '', target);
    connect(target.pathname + target.search);
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('a[data-dashboard-nav]');
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const target = new URL(link.href, location.href);
    history.pushState({}, '', target);
    connect(target.pathname + target.search);
  });
  window.addEventListener('popstate', () => connect(location.pathname + location.search));
  connect(location.pathname + location.search);

})();
