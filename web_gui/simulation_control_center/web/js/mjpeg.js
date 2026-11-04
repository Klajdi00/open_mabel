// MjpegStream — fetch()-based MJPEG reader with a REAL cancel.
//
// WebKit/Safari never aborts a native <img src="…mjpeg"> multipart download:
// not on removeAttribute('src'), not on a data-URI swap, not even when the
// element is removed (measured: all four strategies leaked the connection).
// Every FPV toggle / camera-projection toggle therefore stacked another live
// stream until the browser's 6-per-host pool starved REST calls and new
// streams. fetch() + AbortController cancels deterministically in every
// engine, so consumers that attach/detach within a page's lifetime stream
// through this class instead: it reads the multipart body, extracts JPEG
// frames, and paints them into the target <img> as blob URLs (same-origin,
// so WebGL texturing keeps working; per-frame img.onload keeps firing).
//
// JPEG scan correctness: inside entropy-coded data every real 0xFF byte is
// stuffed as FF 00, so raw FF D8 (SOI) / FF D9 (EOI) can only be markers —
// boundary-format agnostic and safe.

const SOI = 0xd8, EOI = 0xd9;

// pure + exported for tests: pull complete JPEGs out of buf, return the rest
export function extractJpegs(buf){
  const frames = [];
  let start = -1, i = 0;
  while (i < buf.length - 1){
    if (buf[i] === 0xff && buf[i + 1] === SOI && start < 0){ start = i; i += 2; continue; }
    if (buf[i] === 0xff && buf[i + 1] === EOI && start >= 0){
      frames.push(buf.slice(start, i + 2));
      start = -1;
    }
    i++;
  }
  return { frames, rest: buf.slice(start >= 0 ? start : buf.length) };
}

export class MjpegStream {
  constructor(img, opts = {}){
    this.img = img;
    this.onerror = opts.onerror || null;   // network/HTTP failure (not abort)
    this.ac = null;
    this._url = null;                      // last blob URL, revoked on replace
  }

  get running(){ return !!this.ac; }

  attach(url){
    this.stop();
    const ac = new AbortController();
    this.ac = ac;
    this._pump(url, ac);
  }

  stop(){
    this.ac?.abort();
    this.ac = null;
    if (this._url){ URL.revokeObjectURL(this._url); this._url = null; }
  }

  _paint(jpg){
    const url = URL.createObjectURL(new Blob([jpg], { type: 'image/jpeg' }));
    const old = this._url;
    this._url = url;
    this.img.src = url;                    // fires img.onload per frame
    if (old) URL.revokeObjectURL(old);     // safe: decode of `old` is done or dropped
  }

  async _pump(url, ac){
    try {
      // fetch=1 → the proxy retags the stream application/octet-stream:
      // WebKit's fetch() cannot read multipart/x-mixed-replace bodies
      const u = url + (url.includes('?') ? '&' : '?') + 'fetch=1';
      const resp = await fetch(u, { signal: ac.signal, cache: 'no-store' });
      if (!resp.ok || !resp.body) throw new Error('HTTP ' + resp.status);
      const reader = resp.body.getReader();
      let buf = new Uint8Array(0);
      for (;;){
        const { done, value } = await reader.read();
        if (done || ac !== this.ac) break;
        const nb = new Uint8Array(buf.length + value.length);
        nb.set(buf); nb.set(value, buf.length);
        const { frames, rest } = extractJpegs(nb);
        buf = rest.length > 2 ** 22 ? new Uint8Array(0) : rest;   // garbage guard
        if (frames.length) this._paint(frames[frames.length - 1]); // newest only
      }
      // orderly end of stream (camera server restarted) counts as an error so
      // consumers can re-attach — an aborted stream stays silent
      if (ac === this.ac && !ac.signal.aborted) this.onerror?.(new Error('stream ended'));
    } catch (e) {
      if (!ac.signal.aborted) this.onerror?.(e);
    }
  }
}
