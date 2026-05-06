const CACHE = 'yolo-v2';
const SHELL = [
   '/',
   '/index.html',
   '/manifest.json',
   '/icons/icon-192.png',
   '/icons/icon-512.png',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(SHELL))
   );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
     )
   );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/models/') || url.pathname.endsWith('.wasm')) {
    event.respondWith(cacheFirst(event.request));
    return;
   }
   event.respondWith(staleWhileRevalidate(event.request));
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    const clone = response.clone();
    caches.open(CACHE).then(cache => cache.put(request, clone));
   }
   return response;
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fresh = await fetch(request).then(response => {
    if (response.ok) {
      const clone = response.clone();
      caches.open(CACHE).then(cache => cache.put(request, clone));
     }
     return response;
   }).catch(() => cached);
  return fresh || cached || new Response('Network error', { status: 500 });
}
