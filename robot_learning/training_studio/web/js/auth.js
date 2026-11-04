// Operator sign-in for the Trainer Studio — the SAME styled popup as the
// Real-Robot Control Center. The studio browses read-only; the first time you
// take a *mutating* training action (launch / stop / pause / resume / retrain /
// deploy / eval) this popup asks for the operator username + passkey, which the
// server validates into a signed session cookie. ensureAuth() gates each action;
// postAuthed() sends the request and re-prompts once if the session has lapsed.

let _authed = false;                          // in-memory: known-good session?

const CSS = `
.ts-login-veil{position:fixed;inset:0;z-index:10000;display:flex;align-items:center;
  justify-content:center;background:radial-gradient(120% 120% at 50% 0%,rgba(11,14,20,.86) 0%,rgba(5,7,11,.92) 70%);
  -webkit-backdrop-filter:blur(7px);backdrop-filter:blur(7px);
  font-family:'Geist',system-ui,sans-serif;animation:tsFade .35s ease}
@keyframes tsFade{from{opacity:0}to{opacity:1}}
.ts-login-card{width:min(92vw,392px);padding:34px 32px 26px;border-radius:20px;
  background:linear-gradient(180deg,rgba(23,28,38,.95),rgba(15,19,27,.97));
  border:1px solid rgba(255,255,255,.09);
  box-shadow:0 30px 90px -20px rgba(0,0,0,.8),0 0 0 1px rgba(255,255,255,.02) inset;
  color:#e8ecf4}
.ts-login-badge{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;
  justify-content:center;margin:0 auto 18px;font-size:24px;
  background:linear-gradient(160deg,#e5484d,#b02a2f);
  box-shadow:0 8px 24px -6px rgba(229,72,77,.6)}
.ts-login-title{font-family:'Instrument Serif',serif;font-size:26px;line-height:1.1;
  text-align:center;margin:0 0 4px;letter-spacing:.2px}
.ts-login-sub{text-align:center;font-size:12.5px;color:#8b93a4;margin:0 0 22px;letter-spacing:.3px}
.ts-login-field{margin-bottom:12px}
.ts-login-field label{display:block;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;
  color:#7a8394;margin:0 0 6px 2px;font-weight:600}
.ts-login-field input{width:100%;box-sizing:border-box;padding:12px 13px;font-size:14px;color:#f0f3f8;
  background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.1);border-radius:11px;outline:none;
  font-family:inherit;transition:border .15s,background .15s}
.ts-login-field input:focus{border-color:#e5484d;background:rgba(229,72,77,.06)}
.ts-login-btn{width:100%;margin-top:8px;padding:12px;font-size:14px;font-weight:600;letter-spacing:.02em;
  color:#fff;border:none;border-radius:11px;cursor:pointer;background:linear-gradient(160deg,#e5484d,#c02d33);
  transition:filter .15s,transform .05s;font-family:inherit}
.ts-login-btn:hover{filter:brightness(1.08)}
.ts-login-btn:active{transform:translateY(1px)}
.ts-login-btn:disabled{opacity:.6;cursor:progress}
.ts-login-cancel{width:100%;margin-top:8px;padding:9px;font-size:12px;color:#8b93a4;background:none;
  border:none;cursor:pointer;font-family:inherit}
.ts-login-cancel:hover{color:#c3cbd9}
.ts-login-err{min-height:16px;margin:11px 2px 0;font-size:12px;color:#ff7a7f;text-align:center;letter-spacing:.2px}
.ts-login-foot{margin-top:18px;padding-top:14px;border-top:1px solid rgba(255,255,255,.06);text-align:center;
  font-size:10.5px;color:#5f6879;letter-spacing:.3px;line-height:1.5}
.ts-shake{animation:tsShake .4s}
@keyframes tsShake{10%,90%{transform:translateX(-1px)}30%,70%{transform:translateX(-5px)}50%{transform:translateX(5px)}}
`;

function injectCss(){
  if (document.getElementById('ts-login-css')) return;
  const s = document.createElement('style');
  s.id = 'ts-login-css'; s.textContent = CSS; document.head.appendChild(s);
}

async function whoami(){
  try {
    const r = await fetch('/api/whoami', { cache: 'no-store' });
    return r.ok ? await r.json() : { authed: false };
  } catch { return { authed: false }; }
}

function showLoginPopup(action){
  injectCss();
  return new Promise((resolve, reject) => {
    const veil = document.createElement('div');
    veil.className = 'ts-login-veil';
    veil.innerHTML = `
      <form class="ts-login-card" autocomplete="on">
        <div class="ts-login-badge">🔒</div>
        <h1 class="ts-login-title">Operator Sign-In</h1>
        <p class="ts-login-sub">${action ? 'to ' + action : 'training actions'} require your credentials</p>
        <div class="ts-login-field">
          <label for="tsU">Username</label>
          <input id="tsU" name="username" autocomplete="username" autocapitalize="off"
                 autocorrect="off" spellcheck="false" placeholder="username" />
        </div>
        <div class="ts-login-field">
          <label for="tsP">Passkey</label>
          <input id="tsP" name="password" type="password" autocomplete="current-password" placeholder="••••••" />
        </div>
        <button class="ts-login-btn" type="submit">Authorize</button>
        <button class="ts-login-cancel" type="button">Cancel</button>
        <div class="ts-login-err" id="tsErr"></div>
        <div class="ts-login-foot">MABEL Trainer · secured session · required to send or kill training jobs</div>
      </form>`;
    document.body.appendChild(veil);
    const card = veil.querySelector('.ts-login-card');
    const u = veil.querySelector('#tsU'), p = veil.querySelector('#tsP');
    const err = veil.querySelector('#tsErr'), btn = veil.querySelector('.ts-login-btn');
    setTimeout(() => u.focus(), 60);
    const close = () => veil.remove();
    veil.querySelector('.ts-login-cancel').addEventListener('click', () => {
      close(); reject(new Error('sign-in cancelled'));
    });
    card.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = ''; btn.disabled = true; btn.textContent = 'Verifying…';
      try {
        const r = await fetch('/api/login', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: u.value, passkey: p.value }) });
        const j = await r.json().catch(() => ({}));
        if (r.ok && j.ok){ _authed = true; btn.textContent = 'Authorized ✓';
          setTimeout(() => { close(); resolve(); }, 250); return; }
        err.textContent = j.error || 'Login failed';
      } catch { err.textContent = 'Network error — try again'; }
      btn.disabled = false; btn.textContent = 'Authorize';
      card.classList.remove('ts-shake'); void card.offsetWidth;
      card.classList.add('ts-shake'); p.select();
    });
  });
}

// Ensure a valid operator session before a gated action. `action` is a short
// verb phrase shown in the popup subtitle (e.g. "launch this training run").
export async function ensureAuth(action, force){
  if (_authed && !force) return;
  if (!force){
    const who = await whoami();
    if (who.authed){ _authed = true; return; }
  }
  await showLoginPopup(action);
}

// POST that requires an operator session; re-prompts once on a lapsed session.
export async function postAuthed(url, body, action){
  await ensureAuth(action);
  const send = () => fetch(url, { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body || {}) });
  let r = await send();
  if (r.status === 401){ _authed = false; await ensureAuth(action, true); r = await send(); }
  if (!r.ok) throw new Error((await r.text().catch(() => '')) || r.status);
  return r.json();
}
