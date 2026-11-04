// Studio-styled login gate for the Real-Robot Control Center.
// Replaces the browser-native basic-auth prompt with an in-app overlay that
// matches the studio UI and authenticates against /api/login (HttpOnly session
// cookie). requireLogin() resolves only once a valid session exists, so the app
// never boots its data planes for an unauthenticated visitor.

const CSS = `
.cc-login-veil{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;
  justify-content:center;background:radial-gradient(120% 120% at 50% 0%,#0b0e14 0%,#05070b 70%);
  -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);
  font-family:'Geist',system-ui,sans-serif;animation:ccFade .35s ease}
@keyframes ccFade{from{opacity:0}to{opacity:1}}
.cc-login-card{width:min(92vw,392px);padding:34px 32px 26px;border-radius:20px;
  background:linear-gradient(180deg,rgba(23,28,38,.92),rgba(15,19,27,.94));
  border:1px solid rgba(255,255,255,.09);
  box-shadow:0 30px 90px -20px rgba(0,0,0,.8),0 0 0 1px rgba(255,255,255,.02) inset;
  color:#e8ecf4}
.cc-login-badge{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;
  justify-content:center;margin:0 auto 18px;font-size:24px;
  background:linear-gradient(160deg,#e5484d,#b02a2f);
  box-shadow:0 8px 24px -6px rgba(229,72,77,.6)}
.cc-login-title{font-family:'Instrument Serif',serif;font-size:26px;line-height:1.1;
  text-align:center;margin:0 0 4px;letter-spacing:.2px}
.cc-login-sub{text-align:center;font-size:12.5px;color:#8b93a4;margin:0 0 22px;
  letter-spacing:.3px}
.cc-login-field{margin-bottom:12px}
.cc-login-field label{display:block;font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:#7a8394;margin:0 0 6px 2px;font-weight:600}
.cc-login-field input{width:100%;box-sizing:border-box;padding:12px 13px;font-size:14px;
  color:#f0f3f8;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.1);
  border-radius:11px;outline:none;font-family:inherit;transition:border .15s,background .15s}
.cc-login-field input:focus{border-color:#e5484d;background:rgba(229,72,77,.06)}
.cc-login-btn{width:100%;margin-top:8px;padding:12px;font-size:14px;font-weight:600;
  letter-spacing:.02em;color:#fff;border:none;border-radius:11px;cursor:pointer;
  background:linear-gradient(160deg,#e5484d,#c02d33);transition:filter .15s,transform .05s;
  font-family:inherit}
.cc-login-btn:hover{filter:brightness(1.08)}
.cc-login-btn:active{transform:translateY(1px)}
.cc-login-btn:disabled{opacity:.6;cursor:progress}
.cc-login-err{min-height:16px;margin:11px 2px 0;font-size:12px;color:#ff7a7f;
  text-align:center;letter-spacing:.2px}
.cc-login-foot{margin-top:18px;padding-top:14px;border-top:1px solid rgba(255,255,255,.06);
  text-align:center;font-size:10.5px;color:#5f6879;letter-spacing:.3px;line-height:1.5}
.cc-shake{animation:ccShake .4s}
@keyframes ccShake{10%,90%{transform:translateX(-1px)}30%,70%{transform:translateX(-5px)}
  50%{transform:translateX(5px)}}
`;

function injectCss(){
  if (document.getElementById('cc-login-css')) return;
  const s = document.createElement('style');
  s.id = 'cc-login-css'; s.textContent = CSS;
  document.head.appendChild(s);
}

async function whoami(){
  try {
    const r = await fetch('/api/whoami', { cache: 'no-store' });
    if (!r.ok) return { authed: false, required: true };
    return await r.json();
  } catch { return { authed: false, required: true }; }
}

function showLogin(){
  injectCss();
  return new Promise((resolve) => {
    const veil = document.createElement('div');
    veil.className = 'cc-login-veil';
    veil.innerHTML = `
      <form class="cc-login-card" autocomplete="on">
        <div class="cc-login-badge">🔒</div>
        <h1 class="cc-login-title">Real-Robot Control</h1>
        <p class="cc-login-sub">Authorized operators only · sign in to continue</p>
        <div class="cc-login-field">
          <label for="ccU">Username</label>
          <input id="ccU" name="username" autocomplete="username" autocapitalize="off"
                 autocorrect="off" spellcheck="false" placeholder="username" />
        </div>
        <div class="cc-login-field">
          <label for="ccP">Passkey</label>
          <input id="ccP" name="password" type="password" autocomplete="current-password"
                 placeholder="••••••" />
        </div>
        <button class="cc-login-btn" type="submit">Unlock Control Center</button>
        <div class="cc-login-err" id="ccErr"></div>
        <div class="cc-login-foot">MABEL · secured session · robot motion additionally
          gated by the on-robot access code</div>
      </form>`;
    document.body.appendChild(veil);
    const card = veil.querySelector('.cc-login-card');
    const u = veil.querySelector('#ccU'), p = veil.querySelector('#ccP');
    const err = veil.querySelector('#ccErr');
    const btn = veil.querySelector('.cc-login-btn');
    setTimeout(() => u.focus(), 60);

    card.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = ''; btn.disabled = true; btn.textContent = 'Verifying…';
      try {
        const r = await fetch('/api/login', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: u.value, passkey: p.value }) });
        const j = await r.json().catch(() => ({}));
        if (r.ok && j.ok){
          btn.textContent = 'Unlocked ✓';
          veil.style.transition = 'opacity .3s'; veil.style.opacity = '0';
          setTimeout(() => { veil.remove(); resolve(); }, 300);
          return;
        }
        err.textContent = j.error || 'Login failed';
      } catch { err.textContent = 'Network error — try again'; }
      btn.disabled = false; btn.textContent = 'Unlock Control Center';
      card.classList.remove('cc-shake'); void card.offsetWidth;
      card.classList.add('cc-shake'); p.select();
    });
  });
}

// Resolve once authenticated (or auth not required). Blocks app boot otherwise.
export async function requireLogin(){
  const who = await whoami();
  if (who.authed || who.required === false) return who;
  await showLogin();
  return whoami();
}
