"""Live demo: drag a fundus photo into the browser and see the whole pipeline working.

Start with DEMO.bat (double-click). It loads the trained models once, starts a small web server on THIS laptop only
(http://127.0.0.1:8765, nothing leaves the computer, no internet needed) and opens the page in the browser.
Drop / choose a photo -> disc found -> disc, cup and PPA outlined -> CDRs, rim (ISNT), PPA -> glaucoma probability,
paper's rule (area CDR > 0.25), Grad-CAM, image-quality warning. Close the black window to stop.
Research prototype - not a medical device.
"""
import os, sys, json, tempfile, importlib.util, webbrowser, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST, PORT = "127.0.0.1", 8765

print("Loading models (about 20-40 s the first time) ...", flush=True)
spec = importlib.util.spec_from_file_location("predict", os.path.join(ROOT, "scripts", "predict.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = P.Models(DEV)
LOCK = threading.Lock()
print(f"Models ready on {DEV}.", flush=True)

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glaucoma Screening Demo</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d232b;--mute:#5d6875;--line:#e2e6eb;--acc:#2457c5;--bad:#b3261e;--good:#1e7a3c;--warn:#a15c00}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI",system-ui,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:22px 28px;background:#14263f;color:#fff}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:#c9d4e3;font-size:14px}
main{max-width:1150px;margin:0 auto;padding:22px}
#drop{display:block;border:2px dashed #9fb0c6;border-radius:12px;background:var(--card);padding:34px;text-align:center;cursor:pointer;transition:.15s}
#drop.hover{border-color:var(--acc);background:#eef3fd}#drop b{color:var(--acc)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h3{margin:0 0 10px;font-size:15px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
img{max-width:100%;border-radius:8px;display:block}
#verdict{font-size:24px;font-weight:700;margin:0}#prob{font-size:15px;color:var(--mute);margin-top:4px}
.bar{height:12px;border-radius:6px;background:#e7ebf0;overflow:hidden;margin:10px 0 4px}.bar>div{height:100%}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:6px 4px;border-bottom:1px solid var(--line)}td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.steps{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.step{flex:1;min-width:140px;background:#eef1f5;border-radius:8px;padding:8px 10px;font-size:13px;color:var(--mute)}
.step.done{background:#e3f1e7;color:var(--good)}.step.run{background:#e8eefc;color:var(--acc)}
.warn{color:var(--warn)}.legend span{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}
footer{color:var(--mute);font-size:12px;text-align:center;padding:20px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Automatic glaucoma screening from a fundus photo</h1>
<p>Deep-learning re-implementation of Kumar &amp; Kumar (2022): optic disc &rarr; disc &amp; cup &rarr; cup-to-disc ratio &rarr; parapapillary atrophy &rarr; decision. Runs entirely on this computer.</p></header>
<main>
<label id="drop"><input id="file" type="file" accept="image/*" hidden>
<div style="font-size:18px">Drop a fundus photo here or <b>click to choose</b></div>
<div style="color:#5d6875;margin-top:6px;font-size:13px">JPG / PNG / TIF &middot; any camera &middot; try a blurred or dark photo to see the quality warning</div></label>
<div class="steps" id="steps">
<div class="step" id="s1">1. Find optic disc</div><div class="step" id="s2">2. Segment disc &amp; cup</div>
<div class="step" id="s3">3. CDR, rim &amp; PPA</div><div class="step" id="s4">4. CNN + Grad-CAM</div><div class="step" id="s5">5. Decision &amp; quality check</div></div>
<div id="out" style="display:none">
<div class="grid">
 <div class="card"><h3>Input photo</h3><img id="orig"></div>
 <div class="card"><h3>Result</h3><p id="verdict"></p><div id="prob"></div><div class="bar"><div id="pbar"></div></div>
  <div style="font-size:12px;color:#5d6875" id="thr"></div>
  <p id="qual" style="margin-top:12px"></p>
  <p id="rule" style="margin-top:4px;font-size:14px"></p></div>
 <div class="card"><h3>Disc, cup and PPA found</h3><img id="ov">
  <div class="legend" style="font-size:13px;margin-top:8px"><span style="background:#00ffff"></span>disc<span style="background:#ff00ff"></span>cup<span style="background:#ff0000"></span>PPA</div><div id="ppanote" style="font-size:12px;color:#5d6875;margin-top:4px"></div></div>
 <div class="card"><h3>Where the network looked (Grad-CAM)</h3><img id="gc"><div style="font-size:13px;color:#5d6875;margin-top:8px">Red = regions that drove the glaucoma score.</div></div>
 <div class="card"><h3>Measurements</h3><table id="meas"></table></div>
 <div class="card"><h3>Neuroretinal rim (ISNT)</h3><table id="rim"></table><div style="font-size:13px;color:#5d6875;margin-top:8px">Healthy eyes usually follow Inferior &ge; Superior &ge; Nasal &ge; Temporal.</div></div>
</div></div>
<div id="err" class="card" style="display:none;margin-top:18px;color:#b3261e"></div>
</main>
<footer>Research prototype (MNNIT Allahabad). Not a medical device &mdash; any finding must be confirmed by an ophthalmologist.</footer>
<script>
const $=id=>document.getElementById(id), drop=$("drop");
["dragenter","dragover"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add("hover")}));
["dragleave","drop"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove("hover")}));
drop.addEventListener("drop",ev=>{if(ev.dataTransfer.files[0])go(ev.dataTransfer.files[0])});
$("file").addEventListener("change",ev=>{if(ev.target.files[0])go(ev.target.files[0])});
let timer;
function steps(k){for(let i=1;i<=5;i++){$("s"+i).className="step"+(i<k?" done":i==k?" run":"")}}
function row(t,k,v){return `<tr><td>${k}</td><td>${v??"&ndash;"}</td></tr>`}
async function go(f){
  $("err").style.display="none";$("out").style.display="none";$("orig").src=URL.createObjectURL(f);
  let k=1;steps(k);clearInterval(timer);timer=setInterval(()=>{if(k<5)steps(++k)},450);
  try{
    const r=await fetch("/analyse?name="+encodeURIComponent(f.name),{method:"POST",body:f});
    const d=await r.json();clearInterval(timer);
    if(d.error){throw new Error(d.error)}
    steps(6);const R=d.result;$("out").style.display="block";
    if(!d.overlay){$("verdict").textContent=R.warning||"No result";$("verdict").style.color="var(--warn)";return}
    const refer=String(R.decision).startsWith("REFER"), p=R.glaucoma_probability;
    $("verdict").textContent=refer?"Refer: glaucoma suspected":"No glaucoma signs detected";
    $("verdict").style.color=refer?"var(--bad)":"var(--good)";
    $("prob").textContent="Glaucoma probability "+(p*100).toFixed(1)+"%";
    $("pbar").style.width=(p*100)+"%";$("pbar").style.background=refer?"var(--bad)":"var(--good)";
    $("thr").textContent="Referral threshold "+(R.decision_threshold*100).toFixed(1)+"% (set on validation data)";
    const q=R.quality_warning;$("qual").innerHTML=q&&q!=="none"?`<span class="warn">&#9888; Image quality: ${q}. Consider retaking the photo.</span>`:"Image quality: OK";
    $("rule").innerHTML=`Base paper's rule (area CDR &gt; 0.25): <b>${R["paper_rule_area_CDR_gt_0.25"]?"glaucoma":"normal"}</b>`;
    $("ov").src="data:image/png;base64,"+d.overlay;$("ppanote").textContent=(R.PPA_outline&&R.PPA_outline.startsWith("no"))?"No atrophy (PPA) region outlined in this eye - see the ring score in Measurements.":"";$("gc").src="data:image/png;base64,"+d.gradcam;
    $("meas").innerHTML=row(0,"Vertical CDR",R.vCDR)+row(0,"Horizontal CDR",R.hCDR)+row(0,"Area CDR",R.area_CDR)+row(0,"Soft area CDR",R.soft_area_CDR)
      +row(0,"PPA area / disc area",R.PPA_to_disc_area)+row(0,"PPA outline",R.PPA_outline)+row(0,"Peripapillary ring score",R.peripapillary_ring_score!=null?R.peripapillary_ring_score+" ("+R.peripapillary_changes+")":null)+row(0,"Disc-finder confidence",R.localiser_confidence?.toFixed(3))+row(0,"Image size",R.image_size)+row(0,"Time",d.seconds+" s");
    const f3=v=>v==null?null:Number(v).toFixed(3);
    $("rim").innerHTML=row(0,"Inferior",f3(R.rim_I))+row(0,"Superior",f3(R.rim_S))+row(0,"Nasal",f3(R.rim_N))+row(0,"Temporal",f3(R.rim_T))+row(0,"Thinnest rim",f3(R.rim_min))
      +row(0,"ISNT rule followed",R.isnt_rule_ok==null?null:(R.isnt_rule_ok?"yes":"no"));
  }catch(e){clearInterval(timer);steps(0);$("err").style.display="block";$("err").textContent="Could not analyse this image: "+e.message}
}
</script></body></html>"""


def clean(v):
    import numpy as np
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self.send(200, PAGE, "text/html; charset=utf-8") if self.path in ("/", "/index.html") else self.send(404, "not found", "text/plain")

    def do_POST(self):
        if not self.path.startswith("/analyse"):
            return self.send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 60_000_000:
            return self.send(400, json.dumps({"error": "empty or too large file"}), "application/json")
        data = self.rfile.read(n); t0 = time.time()
        fd, tmp = tempfile.mkstemp(suffix=".img"); os.write(fd, data); os.close(fd)
        try:
            with LOCK:
                r, ims = P.analyse(tmp, M)
            from urllib.parse import urlparse, parse_qs
            r["file"] = parse_qs(urlparse(self.path).query).get("name", ["upload"])[0]
            out = {"result": {k: clean(v) for k, v in r.items()}, "seconds": round(time.time() - t0, 2)}
            if ims is not None:
                out["overlay"], out["gradcam"] = P.b64(ims[0]), P.b64(ims[1])
            print(f"  {r['file']}: {r.get('decision', r.get('warning'))}  p={r.get('glaucoma_probability', '-')}  vCDR={r.get('vCDR', '-')}  ({out['seconds']} s)", flush=True)
            self.send(200, json.dumps(out), "application/json")
        except Exception as e:
            self.send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
        finally:
            try: os.remove(tmp)
            except OSError: pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), H)
    url = f"http://{HOST}:{PORT}/"
    print(f"\nDemo running at {url}  (only on this computer). Opening the browser ...\nClose this window to stop.\n", flush=True)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv.serve_forever()
