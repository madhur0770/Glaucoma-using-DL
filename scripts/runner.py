"""Glaucoma project job runner (v2). Start once with START_RUNNER.bat and leave the window open.

Work is queued as job files  jobs\\NN_name.json :
  {"name": "...", "script": "scripts/xyz.py", "args": [...], "requires": ["02_build_master"], "retries": 1}
* jobs run in file-name order, one at a time; output (with progress bars) is shown here AND saved to logs\\jobs\\<job>.txt
* "requires": job ids (file names without .json) that must have finished OK first; if one of them FAILED the job
  waits in the queue (the failed stage is fixed and re-queued, then the dependants continue automatically)
* "retries": automatic re-runs after a failure (for temporary problems such as a network drop), 60 s apart
* finished jobs move to jobs\\done, failed ones to jobs\\failed; status + heartbeat in logs\\runner_status.json
* if this file is updated, the runner restarts itself between jobs (no action needed from you)
* the PC is kept awake while a job runs; only scripts inside the project's scripts\\ folder are allowed.
* after every job: results\\Automatic_Run_Log.xlsx, the research kit and the state\\ backup update by themselves."""
import json, os, subprocess, sys, time, datetime, shutil, glob, ctypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "jobs"); LOGS = os.path.join(ROOT, "logs", "jobs")
DONE, FAILED = os.path.join(JOBS, "done"), os.path.join(JOBS, "failed")
for d in [JOBS, DONE, FAILED, LOGS]:
    os.makedirs(d, exist_ok=True)
STATUS = os.path.join(ROOT, "logs", "runner_status.json")
PY = os.path.join(ROOT, "env", "Scripts", "python.exe")
PY = PY if os.path.exists(PY) else sys.executable
SCRIPTS = os.path.realpath(os.path.join(ROOT, "scripts"))
SELF = os.path.abspath(__file__); SELF_MTIME = os.path.getmtime(SELF)
VERSION = "v3"
history = []
try:
    history = json.load(open(STATUS)).get("history", [])
except Exception:
    pass


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def keep_awake(on):
    try:  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def jid(path):
    return os.path.splitext(os.path.basename(path))[0]


def state_of(job_id):
    if os.path.exists(os.path.join(DONE, job_id + ".json")):
        return "done"
    if os.path.exists(os.path.join(FAILED, job_id + ".json")):
        return "failed"
    if os.path.exists(os.path.join(JOBS, job_id + ".json")):
        return "queued"
    return "missing"


def write_status(state, current=None, waiting=None):
    queue = [jid(j) for j in sorted(glob.glob(os.path.join(JOBS, "*.json")))]
    json.dump({"runner": VERSION, "state": state, "heartbeat": now(), "current": current, "queue": queue,
               "waiting": waiting or {}, "done": sorted(jid(j) for j in glob.glob(os.path.join(DONE, "*.json"))),
               "failed": sorted(jid(j) for j in glob.glob(os.path.join(FAILED, "*.json"))), "history": history[-300:]},
              open(STATUS, "w"), indent=2)


def run_job(jf, job):
    name = job.get("name", jid(jf)); logf = os.path.join(LOGS, jid(jf) + ".txt")
    tries = int(job.get("retries", 1)) + 1
    for attempt in range(1, tries + 1):
        rec = {"job": jid(jf), "name": name, "attempt": attempt, "start": now()}
        print(f"\n>>> [{rec['start']}] START {name}  (attempt {attempt}/{tries})\n", flush=True)
        write_status("running", rec); keep_awake(True); t0 = time.time(); last = 0
        with open(logf, "ab") as lf:
            lf.write(f"\n===== {rec['start']} attempt {attempt}/{tries} =====\n".encode())
            p = subprocess.Popen([PY, "-u", os.path.join(ROOT, job["script"])] + [str(a) for a in job.get("args", [])],
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            while True:
                chunk = p.stdout.read1(4096)
                if not chunk:
                    break
                try:
                    sys.stdout.buffer.write(chunk); sys.stdout.flush()
                except Exception:
                    pass
                lf.write(chunk); lf.flush()
                if time.time() - last > 30:
                    rec["elapsed_min"] = round((time.time() - t0) / 60, 1); write_status("running", rec); last = time.time()
            rc = p.wait()
        keep_awake(False)
        rec.update(end=now(), minutes=round((time.time() - t0) / 60, 1), returncode=rc, log=os.path.relpath(logf, ROOT))
        history.append(rec)
        print(f"\n<<< [{rec['end']}] {'DONE' if rc == 0 else 'FAILED'} {name}  ({rec['minutes']} min, exit code {rc})\n", flush=True)
        if rc == 0:
            return True
        if attempt < tries:
            print("    retrying in 60 s ...", flush=True); write_status("retry-wait", rec); time.sleep(60)
    return False


def housekeeping():
    """after every job: automatic Excel run log + research kit refresh + state backup (scripts/auto_log.py)"""
    hk = os.path.join(ROOT, "scripts", "auto_log.py")
    if os.path.exists(hk):
        try:
            r = subprocess.run([PY, "-u", hk], cwd=ROOT, capture_output=True, text=True, timeout=900)
            print("    [housekeeping] " + " | ".join((r.stdout or r.stderr).strip().splitlines()[-3:]), flush=True)
        except Exception as e:
            print("    [housekeeping] skipped:", e, flush=True)


print("=" * 78)
print(f" GLAUCOMA PROJECT JOB RUNNER {VERSION} - leave this window open (minimise it). Close it to stop.")
print(" Every stage is queued and monitored automatically. Queue folder:", JOBS)
print("=" * 78, flush=True)
while True:
    if os.path.getmtime(SELF) != SELF_MTIME:
        print("\nRunner was updated - restarting ...\n", flush=True)
        os.execv(sys.executable, [sys.executable, "-u", SELF])
    picked, waiting = None, {}
    for jf in sorted(glob.glob(os.path.join(JOBS, "*.json"))):
        try:
            job = json.load(open(jf))
            script = os.path.realpath(os.path.join(ROOT, job["script"]))
            if not script.startswith(SCRIPTS + os.sep) or not os.path.exists(script):
                raise ValueError(f"script not allowed or missing: {job['script']}")
        except Exception as e:
            shutil.move(jf, os.path.join(FAILED, os.path.basename(jf)))
            history.append({"job": jid(jf), "result": f"rejected: {e}", "time": now()}); continue
        deps = {d: state_of(d) for d in job.get("requires", [])}
        if all(s == "done" for s in deps.values()):
            picked = (jf, job); break
        waiting[jid(jf)] = deps
    if picked is None:
        write_status("idle" if not waiting else "waiting", waiting=waiting); time.sleep(15); continue
    jf, job = picked
    ok = run_job(jf, job)
    dest = DONE if ok else FAILED
    for old in [os.path.join(DONE, os.path.basename(jf)), os.path.join(FAILED, os.path.basename(jf))]:
        if os.path.exists(old):
            os.remove(old)
    shutil.move(jf, os.path.join(dest, os.path.basename(jf)))
    write_status("idle")
    housekeeping()
