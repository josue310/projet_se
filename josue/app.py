from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ----------------------------
# Domain model
# ----------------------------
@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = 0

@dataclass
class Slice:
    pid: str
    start: int
    end: int

@dataclass
class Result:
    algorithm: str
    timeline: List[Slice]
    waiting_times: Dict[str, int]
    response_times: Dict[str, int]
    turnaround_times: Dict[str, int]
    avg_waiting: float
    avg_response: float
    avg_turnaround: float

# ----------------------------
# Utilities
# ----------------------------

def dictify_result(r: Result) -> Dict:
    return {
        "algorithm": r.algorithm,
        "timeline": [asdict(s) for s in r.timeline],
        "waiting_times": r.waiting_times,
        "response_times": r.response_times,
        "turnaround_times": r.turnaround_times,
        "avg_waiting": r.avg_waiting,
        "avg_response": r.avg_response,
        "avg_turnaround": r.avg_turnaround,
    }


def compute_metrics(algorithm: str, procs: List[Process], timeline: List[Slice]) -> Result:
    # First response time (time from arrival to first execution)
    first_start: Dict[str, Optional[int]] = {p.pid: None for p in procs}
    finish_time: Dict[str, int] = {p.pid: 0 for p in procs}

    for sl in timeline:
        if first_start[sl.pid] is None:
            first_start[sl.pid] = sl.start
        finish_time[sl.pid] = sl.end

    by_pid = {p.pid: p for p in procs}

    response_times: Dict[str, int] = {}
    waiting_times: Dict[str, int] = {}
    turnaround_times: Dict[str, int] = {}

    # Total run time per process
    run_time: Dict[str, int] = {p.pid: 0 for p in procs}
    for sl in timeline:
        run_time[sl.pid] += sl.end - sl.start

    for p in procs:
        rt = (first_start[p.pid] - p.arrival) if first_start[p.pid] is not None else 0
        tt = finish_time[p.pid] - p.arrival
        wt = tt - p.burst
        response_times[p.pid] = rt
        turnaround_times[p.pid] = tt
        waiting_times[p.pid] = wt

    n = max(1, len(procs))
    return Result(
        algorithm=algorithm,
        timeline=timeline,
        waiting_times=waiting_times,
        response_times=response_times,
        turnaround_times=turnaround_times,
        avg_waiting=sum(waiting_times.values())/n,
        avg_response=sum(response_times.values())/n,
        avg_turnaround=sum(turnaround_times.values())/n,
    )

# ----------------------------
# Scheduling algorithms
# ----------------------------

def fcfs(processes: List[Process]) -> Result:
    t = 0
    timeline: List[Slice] = []
    for p in sorted(processes, key=lambda x: (x.arrival)):
        if t < p.arrival:
            t = p.arrival
        timeline.append(Slice(p.pid, t, t + p.burst))
        t += p.burst
    return compute_metrics("FCFS", processes, timeline)


def sjf_non_preemptive(processes: List[Process]) -> Result:
    t = 0
    ready: List[Process] = []
    procs = sorted(processes, key=lambda p: p.arrival)
    i = 0
    timeline: List[Slice] = []
    while i < len(procs) or ready:
        while i < len(procs) and procs[i].arrival <= t:
            ready.append(procs[i])
            i += 1
        if not ready:
            t = procs[i].arrival
            continue
        # pick shortest burst (tie-breaker: arrival then PID)
        ready.sort(key=lambda p: (p.burst, p.arrival, p.pid))
        p = ready.pop(0)
        timeline.append(Slice(p.pid, t, t + p.burst))
        t += p.burst
    return compute_metrics("SJF (non-preemptive)", processes, timeline)


def priority_non_preemptive(processes: List[Process]) -> Result:
    t = 0
    ready: List[Process] = []
    procs = sorted(processes, key=lambda p: p.arrival)
    i = 0
    timeline: List[Slice] = []
    while i < len(procs) or ready:
        while i < len(procs) and procs[i].arrival <= t:
            ready.append(procs[i])
            i += 1
        if not ready:
            t = procs[i].arrival
            continue
        # Lower number = higher priority by convention
        ready.sort(key=lambda p: (p.priority, p.arrival, p.pid))
        p = ready.pop(0)
        timeline.append(Slice(p.pid, t, t + p.burst))
        t += p.burst
    return compute_metrics("Priority (non-preemptive)", processes, timeline)


def round_robin(processes: List[Process], quantum: int = 2) -> Result:
    t = 0
    queue: List[Tuple[Process, int]] = []  # (proc, remaining)
    procs = sorted(processes, key=lambda p: p.arrival)
    i = 0
    timeline: List[Slice] = []

    while i < len(procs) or queue:
        while i < len(procs) and procs[i].arrival <= t:
            queue.append((procs[i], procs[i].burst))
            i += 1
        if not queue:
            t = procs[i].arrival
            continue
        p, rem = queue.pop(0)
        use = min(quantum, rem)
        start = t
        end = t + use
        timeline.append(Slice(p.pid, start, end))
        t = end
        rem -= use
        # Add any arrivals that happened during this slice
        while i < len(procs) and procs[i].arrival <= t:
            queue.append((procs[i], procs[i].burst))
            i += 1
        if rem > 0:
            queue.append((p, rem))
    return compute_metrics(f"Round Robin (q={quantum})", processes, timeline)

# ----------------------------
# API & UI
# ----------------------------
INDEX_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CPU Scheduling Lab (Flask)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root{--bg:#0f172a;--card:#111827;--muted:#94a3b8;--text:#e5e7eb;--accent:#38bdf8}
    body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu}
    .wrap{max-width:1100px;margin:32px auto;padding:0 16px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .card{background:var(--card);border-radius:16px;padding:16px;box-shadow:0 10px 20px rgba(0,0,0,.25)}
    textarea, input, select, button{width:100%;padding:10px;border-radius:10px;border:1px solid #1f2937;background:#0b1220;color:var(--text)}
    button{background:var(--accent);color:#002433;font-weight:700;border:none;cursor:pointer}
    button:disabled{opacity:.5;cursor:not-allowed}
    table{width:100%;border-collapse:collapse}
    th, td{padding:8px;border-bottom:1px solid #1f2937;text-align:left}
    .muted{color:var(--muted)}
    .row{display:flex;gap:8px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>🧪 CPU Scheduling Lab</h1>
    <p class="muted">Saisis tes processus, choisis un algorithme et compare les métriques (attente, réponse, turnaround). Gantt inclus.</p>
    <div class="grid">
      <div class="card">
        <h2>Entrée</h2>
        <label>Processus (JSON) :</label>
        <textarea id="procs" rows="10">[
  {"pid":"P1","arrival":0,"burst":7,"priority":2},
  {"pid":"P2","arrival":2,"burst":4,"priority":1},
  {"pid":"P3","arrival":4,"burst":1,"priority":3},
  {"pid":"P4","arrival":5,"burst":4,"priority":2}
]</textarea>
        <div class="row">
          <div style="flex:1">
            <label>Algorithme :</label>
            <select id="algo">
              <option value="FCFS">FCFS</option>
              <option value="SJF">SJF (non-preemptive)</option>
              <option value="PRIORITY">Priority (non-preemptive)</option>
              <option value="RR">Round Robin</option>
            </select>
          </div>
          <div style="flex:.8">
            <label>Quantum (RR) :</label>
            <input id="quantum" type="number" value="2" min="1" />
          </div>
        </div>
        <div class="row">
          <button id="run">Lancer</button>
          <button id="compare">Comparer tous</button>
        </div>
        <p class="muted">Format attendu: [{"pid":"P1","arrival":0,"burst":5,"priority":1}, ...]</p>
      </div>

      <div class="card">
        <h2>Gantt</h2>
        <canvas id="gantt" height="160"></canvas>
        <div id="legend" class="muted" style="margin-top:8px"></div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>Métriques</h2>
      <div id="metrics"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>Comparaison (avg)</h2>
      <canvas id="bar" height="200"></canvas>
    </div>
  </div>
<script>
let ganttChart = null; let barChart = null;

function palette(n){
  // generate n distinct HSL colors (no custom colors to keep it simple/changeable)
  return Array.from({length:n}, (_,i)=>`hsl(${(i* 360/n)|0} 70% 55%)`);
}

function drawGantt(timeline){
  const ctx = document.getElementById('gantt').getContext('2d');
  const maxT = timeline.reduce((m,s)=>Math.max(m,s.end),0);
  const pids = [...new Set(timeline.map(s=>s.pid))];
  const colors = Object.fromEntries(pids.map((p,i)=>[p, palette(pids.length)[i]]));
  const data = {
    labels: Array.from({length:maxT},(_,i)=>i),
    datasets: pids.map(pid=>({
      label: pid,
      data: Array.from({length:maxT},()=>0),
      borderWidth: 8,
      borderColor: colors[pid],
      pointRadius: 0,
      showLine: true,
      parsing:false
    }))
  };
  // mark active segments by raising to y index
  const yIndex = Object.fromEntries(pids.map((p,i)=>[p, i+1]));
  for(const sl of timeline){
    const ds = data.datasets[pids.indexOf(sl.pid)];
    for(let t=sl.start;t<=sl.end;t++){
      if (t < data.labels.length) ds.data[t] = yIndex[sl.pid];
    }
  }
  if(ganttChart) ganttChart.destroy();
  ganttChart = new Chart(ctx,{
    type:'line',
    data,
    options:{
      animation:false,
      plugins:{legend:{display:true}},
      scales:{
        x:{title:{display:true,text:'Temps'}},
        y:{
          suggestedMin:0,suggestedMax:pids.length+1,
          ticks:{callback:(v)=> pids[v-1]||''},
          title:{display:true,text:'Processus'}
        }
      },
      elements:{line:{tension:0},point:{radius:0}}
    }
  });
  document.getElementById('legend').textContent = `Durée totale: ${maxT}`;
}

function renderMetrics(res){
  const m = res;
  const tbl = [
    '<table><thead><tr><th>PID</th><th>Waiting</th><th>Response</th><th>Turnaround</th></tr></thead><tbody>',
    ...Object.keys(m.waiting_times).map(pid=>`<tr><td>${pid}</td><td>${m.waiting_times[pid]}</td><td>${m.response_times[pid]}</td><td>${m.turnaround_times[pid]}</td></tr>`),
    `</tbody></table>
    <p>Moyennes → Attente: <b>${m.avg_waiting.toFixed(2)}</b> · Réponse: <b>${m.avg_response.toFixed(2)}</b> · Turnaround: <b>${m.avg_turnaround.toFixed(2)}</b></p>`
  ].join('');
  document.getElementById('metrics').innerHTML = tbl;
}

async function simulate(kind){
  const procs = JSON.parse(document.getElementById('procs').value);
  const algo = document.getElementById('algo').value;
  const q = Number(document.getElementById('quantum').value)||2;
  const resp = await fetch('/simulate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({procs, algo, quantum:q, kind})});
  return resp.json();
}

document.getElementById('run').onclick = async ()=>{
  const {result} = await simulate('one');
  drawGantt(result.timeline);
  renderMetrics(result);
  // update bar chart with this single algo
  drawBars([{name: result.algorithm, m: result}]);
};

function drawBars(items){
  const ctx = document.getElementById('bar').getContext('2d');
  const labels = items.map(x=>x.name);
  const avgW = items.map(x=>x.m.avg_waiting);
  const avgR = items.map(x=>x.m.avg_response);
  const avgT = items.map(x=>x.m.avg_turnaround);
  if(barChart) barChart.destroy();
  barChart = new Chart(ctx,{
    type:'bar',
    data:{
      labels,
      datasets:[
        {label:'Avg Waiting', data:avgW},
        {label:'Avg Response', data:avgR},
        {label:'Avg Turnaround', data:avgT}
      ]
    },
    options:{responsive:true, animation:false}
  });
}

document.getElementById('compare').onclick = async ()=>{
  const {results} = await simulate('all');
  // show first on Gantt
  drawGantt(results[0].timeline);
  renderMetrics(results[0]);
  drawBars(results.map(r=>({name:r.algorithm, m:r})));
};
</script>
</body>
</html>
"""


# ----------------------------
# Flask routes
# ----------------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


def parse_processes(items: List[Dict]) -> List[Process]:
    procs: List[Process] = []
    for it in items:
        procs.append(Process(
            pid=str(it["pid"]),
            arrival=int(it.get("arrival", 0)),
            burst=int(it.get("burst", 1)),
            priority=int(it.get("priority", 0)),
        ))
    # sanity: sort by arrival then pid
    procs.sort(key=lambda p: (p.arrival, p.pid))
    return procs


@app.post("/simulate")
def simulate_api():
    data = request.get_json(force=True)
    procs = parse_processes(data.get("procs", []))
    algo = data.get("algo", "FCFS")
    quantum = int(data.get("quantum", 2))
    kind = data.get("kind", "one")

    def run_one(name:str) -> Result:
        if name == "FCFS":
            return fcfs(procs)
        if name == "SJF":
            return sjf_non_preemptive(procs)
        if name == "PRIORITY":
            return priority_non_preemptive(procs)
        if name == "RR":
            return round_robin(procs, quantum=max(1, quantum))
        return fcfs(procs)

    if kind == "all":
        algos = ["FCFS", "SJF", "PRIORITY", "RR"]
        results = [dictify_result(run_one(a)) for a in algos]
        return jsonify({"results": results})

    result = dictify_result(run_one(algo))
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
