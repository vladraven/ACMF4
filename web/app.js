class DashboardApp {
  constructor() {
    this.data = null;
    this.syntheticData = window.ACMF_DATA || null;

    this.tsCanvas = document.getElementById('timeSeriesCanvas');
    this.tsCtx = this.tsCanvas.getContext('2d');
    this.phaseCanvas = document.getElementById('phasePortraitCanvas');
    this.phaseCtx = this.phaseCanvas.getContext('2d');

    this.seriesMode = 'deficits';
    this.phasePlane = 'sid1_inst';

    this.colors = {
      sid_1: '#bf616a',
      sid_2: '#ebcb8b',
      sid_3: '#88c0d0',
      inst: '#5e81ac',
      prod: '#b48ead',
      ch: '#a3be8c',
      m: '#8fbcbb',
      scar: '#d08770',
      rec_debt: '#e5e9f0',
      f: '#4c566a',
      variance: '#bf616a',
      ar1: '#ebcb8b',
    };

    this.initEventListeners();
    this.resizeCanvas();
    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.render();
    });

    // Автозагрузка данных из data.js при старте
    if (this.syntheticData && this.syntheticData.tests && this.syntheticData.tests.length > 0) {
      document.getElementById('synthetic-scenario-select').value = this.syntheticData.tests[0].id;
      this.loadScenario(this.syntheticData.tests[0]);
    }
  }

  resizeCanvas() {
    [this.tsCanvas, this.phaseCanvas].forEach(c => {
      const rect = c.parentElement.getBoundingClientRect();
      c.width = rect.width;
      c.height = rect.height;
    });
  }

  initEventListeners() {
    document.getElementById('synthetic-scenario-select').addEventListener('change', e => {
      const testId = e.target.value;
      if (this.syntheticData && testId) {
        const found = this.syntheticData.tests.find(t => t.id === testId);
        if (found) this.loadScenario(found);
      }
    });

    const fileInput = document.getElementById('json-file-input');
    document.getElementById('load-file-btn').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = evt => {
          const parsed = JSON.parse(evt.target.result);
          if (parsed.tests && parsed.tests.length > 0) {
            this.syntheticData = parsed;
            document.getElementById('synthetic-scenario-select').value = parsed.tests[0].id;
            this.loadScenario(parsed.tests[0]);
          } else {
            this.loadScenario(parsed);
          }
        };
        reader.readAsText(file);
      }
    });

    document.getElementById('series-mode-select').addEventListener('change', e => {
      this.seriesMode = e.target.value;
      this.render();
    });

    document.getElementById('phase-plane-select').addEventListener('change', e => {
      this.phasePlane = e.target.value;
      this.render();
    });
  }

  loadScenario(payload) {
    this.data = payload;
    document.getElementById('val-scenario').innerText = payload.name || payload.metadata?.scenario || 'Custom';
    document.getElementById('val-steps').innerText = payload.times?.length || 0;

    const statusEl = document.getElementById('val-test-status');
    if (payload.passed !== undefined) {
      statusEl.innerText = payload.passed ? 'PASSED' : 'FAILED';
      statusEl.style.color = payload.passed ? '#a3be8c' : '#bf616a';
    } else {
      statusEl.innerText = 'SIMULATED';
      statusEl.style.color = '#88c0d0';
    }

    this.render();
  }

  render() {
    if (!this.data) return;
    this.renderTimeSeries();
    this.renderPhasePortrait();
  }

  renderTimeSeries() {
    const ctx = this.tsCtx;
    const w = this.tsCanvas.width;
    const h = this.tsCanvas.height;
    const pad = 40;

    ctx.clearRect(0, 0, w, h);
    const times = this.data.times;
    let seriesToPlot = [];

    if (this.seriesMode === 'deficits') {
      seriesToPlot = [
        { key: 'sid_1', label: 'SID 1', data: this.data.states.sid_1, color: this.colors.sid_1 },
        { key: 'sid_2', label: 'SID 2', data: this.data.states.sid_2, color: this.colors.sid_2 },
        { key: 'sid_3', label: 'SID 3', data: this.data.states.sid_3, color: this.colors.sid_3 },
      ];
    } else if (this.seriesMode === 'capacities') {
      seriesToPlot = [
        { key: 'inst', label: 'Inst', data: this.data.states.inst, color: this.colors.inst },
        { key: 'prod', label: 'Prod', data: this.data.states.prod, color: this.colors.prod },
        { key: 'ch', label: 'Ch', data: this.data.states.ch, color: this.colors.ch },
        { key: 'm', label: 'M', data: this.data.states.m, color: this.colors.m },
      ];
    } else if (this.seriesMode === 'memory') {
      seriesToPlot = [
        { key: 'scar', label: 'Scar', data: this.data.states.scar, color: this.colors.scar },
        { key: 'rec_debt', label: 'RecDebt', data: this.data.states.rec_debt, color: this.colors.rec_debt },
      ];
    } else if (this.seriesMode === 'ews' && this.data.ews) {
      seriesToPlot = [
        { key: 'variance', label: 'Var(Z)', data: this.data.ews.variance || [], color: this.colors.variance },
        { key: 'ar1', label: 'AR(1)', data: this.data.ews.ar1 || [], color: this.colors.ar1 },
      ];
    }

    this.updateLegend(seriesToPlot);
    if (seriesToPlot.length === 0 || !seriesToPlot[0].data || seriesToPlot[0].data.length === 0) return;

    let minY = Infinity, maxY = -Infinity;
    seriesToPlot.forEach(s => {
      s.data.forEach(v => {
        if (v < minY) minY = v;
        if (v > maxY) maxY = v;
      });
    });

    minY = Math.min(minY, 0.0);
    maxY = Math.max(maxY, 1.0);
    const rangeY = (maxY - minY) || 1.0;
    const minX = times[0];
    const maxX = times[times.length - 1];
    const rangeX = (maxX - minX) || 1.0;

    // Сетка
    ctx.strokeStyle = '#2e3440';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.stroke();

    // Линии
    seriesToPlot.forEach(s => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      s.data.forEach((val, i) => {
        const x = pad + ((times[i] - minX) / rangeX) * (w - 2 * pad);
        const y = h - pad - ((val - minY) / rangeY) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  renderPhasePortrait() {
    const ctx = this.phaseCtx;
    const w = this.phaseCanvas.width;
    const h = this.phaseCanvas.height;
    const pad = 40;

    ctx.clearRect(0, 0, w, h);
    if (!this.data.phase_space) return;

    let points = [];
    if (this.phasePlane === 'sid1_inst') points = this.data.phase_space.sid_1_vs_inst;
    else if (this.phasePlane === 'sid2_inst') points = this.data.phase_space.sid_2_vs_inst;
    else if (this.phasePlane === 'sid1_sid2') points = this.data.phase_space.sid_1_vs_sid_2;
    else if (this.phasePlane === 'f_prod') points = this.data.phase_space.f_vs_prod;

    if (!points || points.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(pt => {
      if (pt[0] < minX) minX = pt[0];
      if (pt[0] > maxX) maxX = pt[0];
      if (pt[1] < minY) minY = pt[1];
      if (pt[1] > maxY) maxY = pt[1];
    });

    const rangeX = (maxX - minX) || 1.0;
    const rangeY = (maxY - minY) || 1.0;

    // Сетка
    ctx.strokeStyle = '#2e3440';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.stroke();

    // Траектория
    ctx.strokeStyle = '#5e81ac';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    points.forEach((pt, i) => {
      const x = pad + ((pt[0] - minX) / rangeX) * (w - 2 * pad);
      const y = h - pad - ((pt[1] - minY) / rangeY) * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Старт (зеленый)
    const startX = pad + ((points[0][0] - minX) / rangeX) * (w - 2 * pad);
    const startY = h - pad - ((points[0][1] - minY) / rangeY) * (h - 2 * pad);
    ctx.fillStyle = '#a3be8c';
    ctx.beginPath();
    ctx.arc(startX, startY, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Конец (красный)
    const endX = pad + ((points[points.length - 1][0] - minX) / rangeX) * (w - 2 * pad);
    const endY = h - pad - ((points[points.length - 1][1] - minY) / rangeY) * (h - 2 * pad);
    ctx.fillStyle = '#bf616a';
    ctx.beginPath();
    ctx.arc(endX, endY, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  updateLegend(series) {
    const el = document.getElementById('timeseries-legend');
    el.innerHTML = series
      .map(s => `<div class="legend-item"><div class="legend-color" style="background: ${s.color};"></div> ${s.label}</div>`)
      .join('');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new DashboardApp();
});