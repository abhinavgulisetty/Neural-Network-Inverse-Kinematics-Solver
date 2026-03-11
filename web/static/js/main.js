/* ===== Neural IK Solver Dashboard — JavaScript ===== */

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    loadIterations();
    setupSliders();
    initArm3D();
    initTrajPlot();

    // Modal Logic
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const closeBtn = document.querySelector('.close-modal');

    document.querySelectorAll('.plot-img').forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = "block";
            modalImg.src = this.src;
        });
    });

    closeBtn.addEventListener('click', () => modal.style.display = "none");
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = "none";
    });

});

// ===== API Helpers =====
async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

// ===== Metric Cards =====
async function loadMetrics() {
    try {
        const data = await fetchJSON('/api/metrics');
        if (data.error) { setStatus('No data yet', false); return; }

        const best = data.best_iteration;
        const iters = data.iterations || [];
        const bestMetrics = iters.find(it => it.iteration === best);
        const baseline = data.numerical_baseline;

        if (bestMetrics) {
            animateValue('posRmse', bestMetrics.position_rmse_mm, 4);
            animateValue('oriRmse', bestMetrics.orientation_rmse_deg, 4);
            animateValue('inferenceTime', bestMetrics.avg_inference_ms, 4);
            document.getElementById('bestIter').textContent = best;

            // Bars
            setBar('posBar', Math.min(100, (1 / Math.max(bestMetrics.position_rmse_mm, 0.01)) * 100));
            setBar('oriBar', Math.min(100, (0.5 / Math.max(bestMetrics.orientation_rmse_deg, 0.01)) * 100));
            setBar('inferBar', Math.min(100, (1 / Math.max(bestMetrics.avg_inference_ms, 0.001)) * 100));
            setBar('iterBar', (best / 5) * 100);

            // Color code metrics
            colorMetric('posRmse', bestMetrics.position_rmse_mm <= 1.0);
            colorMetric('oriRmse', bestMetrics.orientation_rmse_deg <= 0.5);
            colorMetric('inferenceTime', bestMetrics.avg_inference_ms <= 1.0);
        }

        if (data.speedup_factor) {
            animateValue('speedupFactor', data.speedup_factor, 0);
            setBar('speedupBar', Math.min(100, (data.speedup_factor / 1000) * 100));
        }

        // Comparison table
        if (bestMetrics && baseline) {
            buildComparisonTable(bestMetrics, baseline, best);
        }

        setStatus('Model Ready', true);
    } catch (e) {
        console.error('Failed to load metrics:', e);
        setStatus('Metrics unavailable', false);
    }
}

function animateValue(id, target, decimals) {
    const el = document.getElementById(id);
    if (!el) return;
    const duration = 1200;
    const start = performance.now();
    const initial = 0;

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = initial + (target - initial) * eased;
        el.textContent = current.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function setBar(id, pct) {
    const el = document.getElementById(id);
    if (el) setTimeout(() => { el.style.width = Math.min(100, Math.max(0, pct)) + '%'; }, 200);
}

function colorMetric(id, isGood) {
    const el = document.getElementById(id);
    if (el) el.style.color = isGood ? '#22c55e' : '#f97316';
}

function setStatus(text, ok) {
    document.getElementById('statusText').textContent = text;
    const badge = document.getElementById('statusBadge');
    badge.style.borderColor = ok ? 'rgba(34,197,94,0.3)' : 'rgba(249,115,22,0.3)';
    badge.style.background = ok ? 'rgba(34,197,94,0.1)' : 'rgba(249,115,22,0.1)';
    badge.style.color = ok ? '#22c55e' : '#f97316';
    badge.querySelector('.status-dot').style.background = ok ? '#22c55e' : '#f97316';
}

// ===== Comparison Table =====
function buildComparisonTable(nn, num, bestIter) {
    const rows = [
        ['Avg Solve Time', `${nn.avg_inference_ms.toFixed(4)} ms`, `${num.avg_solve_time_ms.toFixed(2)} ms`, nn.avg_inference_ms < num.avg_solve_time_ms ? 'nn' : 'num'],
        ['Position RMSE', `${nn.position_rmse_mm.toFixed(4)} mm`, `${num.position_rmse_mm.toFixed(4)} mm`, nn.position_rmse_mm < num.position_rmse_mm ? 'nn' : 'num']
    ];
    const body = document.getElementById('comparisonBody');
    body.innerHTML = rows.map(([metric, nnVal, numVal, winner]) =>
        `<tr>
            <td>${metric}</td>
            <td class="${winner === 'nn' ? 'winner-nn' : ''}">${nnVal}</td>
            <td class="${winner === 'num' ? 'winner-num' : ''}">${numVal}</td>
            <td class="${winner === 'nn' ? 'winner-nn' : 'winner-num'}">${winner === 'nn' ? 'NN' : 'Numerical'}</td>
        </tr>`
    ).join('');
}

// ===== Iteration Cards =====
async function loadIterations() {
    try {
        const data = await fetchJSON('/api/iterations');
        const container = document.getElementById('iterationCards');
        const iters = data.iterations || [];
        const bestIter = data.best_model?.iteration;

        if (iters.length === 0) {
            container.innerHTML = '<div class="glass-card loading-card"><p>No training iterations completed yet.</p></div>';
            return;
        }

        container.innerHTML = iters.map(it => {
            const isBest = it.iteration === bestIter;
            const m = it.metrics || {};
            return `
                <div class="iter-card ${isBest ? 'best' : ''}">
                    <div class="iter-header">Iteration ${it.iteration}</div>
                    <div class="iter-arch">${it.architecture || 'Unknown'}</div>
                    <div class="iter-metrics">
                        <span class="iter-metric-label">Pos RMSE:</span>
                        <span class="iter-metric-val">${(m.position_rmse_mm || 0).toFixed(3)} mm</span>
                        <span class="iter-metric-label">Ori RMSE:</span>
                        <span class="iter-metric-val">${(m.orientation_rmse_deg || 0).toFixed(3)}°</span>
                        <span class="iter-metric-label">Inference:</span>
                        <span class="iter-metric-val">${(m.avg_inference_ms || 0).toFixed(4)} ms</span>
                    </div>
                    <div class="iter-changes">${it.changes_made || 'No notes'}</div>
                </div>`;
        }).join('');
    } catch (e) {
        console.error('Failed to load iterations:', e);
    }
}

// ===== Slider Setup =====
function setupSliders() {
    const sliders = ['inputX', 'inputY', 'inputZ', 'inputRoll', 'inputPitch', 'inputYaw'];
    const vals = ['valX', 'valY', 'valZ', 'valRoll', 'valPitch', 'valYaw'];
    sliders.forEach((sid, i) => {
        const slider = document.getElementById(sid);
        const val = document.getElementById(vals[i]);
        if (slider && val) {
            slider.addEventListener('input', () => {
                val.textContent = parseFloat(slider.value).toFixed(2);
            });
        }
    });
}

// ===== 3D Arm Visualization =====
const plotlyDarkLayout = {
    paper_bgcolor: 'rgba(12,12,24,0)',
    plot_bgcolor: 'rgba(12,12,24,0)',
    font: { color: '#94a3b8', family: 'Inter' },
    scene: {
        xaxis: { color: '#64748b', gridcolor: 'rgba(255,255,255,0.05)', title: 'X (m)' },
        yaxis: { color: '#64748b', gridcolor: 'rgba(255,255,255,0.05)', title: 'Y (m)' },
        zaxis: { color: '#64748b', gridcolor: 'rgba(255,255,255,0.05)', title: 'Z (m)' },
        bgcolor: 'rgba(12,12,24,0)',
        aspectmode: 'cube',
    },
    margin: { l: 0, r: 0, t: 30, b: 0 },
};

function initArm3D() {
    Plotly.newPlot('armPlot3D', [], {
        ...plotlyDarkLayout,
        title: { text: 'Waiting for prediction...', font: { color: '#64748b', size: 13 } },
    }, { responsive: true, displayModeBar: false });
}

function drawArm3D(divId, armPositions, targetPos, title) {
    if (!armPositions || armPositions.length === 0) return;

    const xs = armPositions.map(p => p[0]);
    const ys = armPositions.map(p => p[1]);
    const zs = armPositions.map(p => p[2]);

    const traces = [
        // Links
        {
            x: xs, y: ys, z: zs, type: 'scatter3d', mode: 'lines+markers',
            line: { color: '#00d4ff', width: 8 },
            marker: { size: 6, color: '#8b5cf6' },
            name: 'Robot Arm'
        },
        // Base
        {
            x: [xs[0]], y: [ys[0]], z: [zs[0]], type: 'scatter3d', mode: 'markers',
            marker: { size: 10, color: '#22c55e', symbol: 'diamond' },
            name: 'Base'
        },
        // End effector
        {
            x: [xs[xs.length - 1]], y: [ys[ys.length - 1]], z: [zs[zs.length - 1]],
            type: 'scatter3d', mode: 'markers',
            marker: { size: 10, color: '#ef4444' },
            name: 'End Effector'
        },
    ];

    if (targetPos) {
        traces.push({
            x: [targetPos[0]], y: [targetPos[1]], z: [targetPos[2]],
            type: 'scatter3d', mode: 'markers',
            marker: { size: 8, color: '#f97316', symbol: 'x' },
            name: 'Target'
        });
    }

    Plotly.react(divId, traces, {
        ...plotlyDarkLayout,
        title: { text: title || '', font: { color: '#94a3b8', size: 13 } },
        showlegend: true,
        legend: { font: { color: '#94a3b8' }, bgcolor: 'rgba(0,0,0,0)' },
    });
}

// ===== Prediction =====
async function predict() {
    const x = document.getElementById('inputX').value;
    const y = document.getElementById('inputY').value;
    const z = document.getElementById('inputZ').value;
    const roll = document.getElementById('inputRoll').value;
    const pitch = document.getElementById('inputPitch').value;
    const yaw = document.getElementById('inputYaw').value;

    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = '<p style="color:#00d4ff">Computing...</p>';

    try {
        const data = await fetchJSON(`/api/predict?x=${x}&y=${y}&z=${z}&roll=${roll}&pitch=${pitch}&yaw=${yaw}`);
        if (data.error) { resultDiv.innerHTML = `<p style="color:#ef4444">Error: ${data.error}</p>`; return; }

        const posColor = data.position_error_mm < 1.0 ? '#22c55e' : '#f97316';
        resultDiv.innerHTML = `
            <div style="margin-bottom:0.5rem;color:#94a3b8;">Joint Angles (deg):</div>
            <div style="color:#00d4ff;margin-bottom:0.8rem;">
                [${data.joint_angles_deg.map(v => v.toFixed(2)).join(', ')}]
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;">
                <span style="color:#94a3b8;">Position Error:</span>
                <span style="color:${posColor};text-align:right;">${data.position_error_mm.toFixed(4)} mm</span>
                <span style="color:#94a3b8;">Orient. Error:</span>
                <span style="color:#8b5cf6;text-align:right;">${data.orientation_error_deg.toFixed(4)}°</span>
                <span style="color:#94a3b8;">Inference Time:</span>
                <span style="color:#f97316;text-align:right;">${data.inference_time_ms.toFixed(4)} ms</span>
            </div>`;

        drawArm3D('armPlot3D', data.arm_positions, [parseFloat(x), parseFloat(y), parseFloat(z)],
            `Error: ${data.position_error_mm.toFixed(3)} mm | ${data.inference_time_ms.toFixed(3)} ms`);
    } catch (e) {
        resultDiv.innerHTML = `<p style="color:#ef4444">Request failed: ${e.message}</p>`;
    }
}

async function randomDemo() {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = '<p style="color:#00d4ff">Generating random pose...</p>';

    try {
        const data = await fetchJSON('/api/random-demo');
        if (data.error) { resultDiv.innerHTML = `<p style="color:#ef4444">Error: ${data.error}</p>`; return; }

        // Update sliders
        if (data.target_pose) {
            const ids = ['inputX', 'inputY', 'inputZ', 'inputRoll', 'inputPitch', 'inputYaw'];
            const vids = ['valX', 'valY', 'valZ', 'valRoll', 'valPitch', 'valYaw'];
            data.target_pose.forEach((v, i) => {
                const slider = document.getElementById(ids[i]);
                const valEl = document.getElementById(vids[i]);
                if (slider) slider.value = v;
                if (valEl) valEl.textContent = v.toFixed(2);
            });
        }

        const posColor = data.position_error_mm < 1.0 ? '#22c55e' : '#f97316';
        resultDiv.innerHTML = `
            <div style="margin-bottom:0.5rem;color:#94a3b8;">Predicted Joint Angles (deg):</div>
            <div style="color:#00d4ff;margin-bottom:0.5rem;">
                [${data.joint_angles_deg.map(v => v.toFixed(2)).join(', ')}]
            </div>
            <div style="margin-bottom:0.5rem;color:#94a3b8;">Ground Truth (deg):</div>
            <div style="color:#8b5cf6;margin-bottom:0.8rem;">
                [${data.ground_truth_joints_deg.map(v => v.toFixed(2)).join(', ')}]
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;">
                <span style="color:#94a3b8;">Position Error:</span>
                <span style="color:${posColor};text-align:right;">${data.position_error_mm.toFixed(4)} mm</span>
                <span style="color:#94a3b8;">Orient. Error:</span>
                <span style="color:#8b5cf6;text-align:right;">${data.orientation_error_deg.toFixed(4)}°</span>
                <span style="color:#94a3b8;">Inference:</span>
                <span style="color:#f97316;text-align:right;">${data.inference_time_ms.toFixed(4)} ms</span>
            </div>`;

        const tp = data.target_pose;
        drawArm3D('armPlot3D', data.arm_positions, tp ? tp.slice(0, 3) : null,
            `Error: ${data.position_error_mm.toFixed(3)} mm`);
    } catch (e) {
        resultDiv.innerHTML = `<p style="color:#ef4444">Request failed: ${e.message}</p>`;
    }
}

// ===== Trajectory =====
let trajAnimFrame = 0;
let trajAnimInterval = null;

function initTrajPlot() {
    Plotly.newPlot('trajPlot3D', [], {
        ...plotlyDarkLayout,
        title: { text: 'Select a trajectory and click Run', font: { color: '#64748b', size: 13 } },
    }, { responsive: true, displayModeBar: false });
}

async function runTrajectory() {
    const type = document.getElementById('trajType').value;
    const btn = document.getElementById('trajBtn');
    const statsDiv = document.getElementById('trajStats');

    btn.disabled = true;
    btn.textContent = 'Computing...';
    statsDiv.innerHTML = '<p style="color:#00d4ff">Running trajectory IK...</p>';

    // Update GIF
    const gifImg = document.getElementById('trajGif');
    if (gifImg) {
        gifImg.src = `/static/plots/arm_animation_${type}.gif`;
        gifImg.onerror = () => { gifImg.style.display = 'none'; };
        gifImg.onload = () => { gifImg.style.display = 'block'; };
    }

    try {
        const data = await fetchJSON(`/api/trajectory?type=${type}&points=60`);
        if (data.error) {
            statsDiv.innerHTML = `<p style="color:#ef4444">Error: ${data.error}</p>`;
            btn.disabled = false; btn.textContent = '▶ Run Trajectory';
            return;
        }

        statsDiv.innerHTML = `
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem;font-family:'JetBrains Mono',monospace;">
                <span style="color:#94a3b8;">Type:</span>
                <span style="color:#00d4ff;text-align:right;">${data.type}</span>
                <span style="color:#94a3b8;">Points:</span>
                <span style="color:#00d4ff;text-align:right;">${data.n_points}</span>
                <span style="color:#94a3b8;">Avg Error:</span>
                <span style="color:#22c55e;text-align:right;">${data.avg_error_mm.toFixed(3)} mm</span>
                <span style="color:#94a3b8;">Max Error:</span>
                <span style="color:#f97316;text-align:right;">${data.max_error_mm.toFixed(3)} mm</span>
            </div>`;

        // Animate trajectory
        if (trajAnimInterval) clearInterval(trajAnimInterval);
        trajAnimFrame = 0;
        const frames = data.frames;
        const eeTrace = [];

        trajAnimInterval = setInterval(() => {
            if (trajAnimFrame >= frames.length) {
                trajAnimFrame = 0;
                eeTrace.length = 0;
            }

            const frame = frames[trajAnimFrame];
            const arm = frame.arm_positions;
            eeTrace.push(arm[arm.length - 1]);

            const xs = arm.map(p => p[0]);
            const ys = arm.map(p => p[1]);
            const zs = arm.map(p => p[2]);

            const traces = [
                {
                    x: xs, y: ys, z: zs, type: 'scatter3d', mode: 'lines+markers',
                    line: { color: '#00d4ff', width: 8 }, marker: { size: 5, color: '#8b5cf6' },
                    name: 'Arm', showlegend: trajAnimFrame === 0
                },
                {
                    x: [xs[0]], y: [ys[0]], z: [zs[0]], type: 'scatter3d', mode: 'markers',
                    marker: { size: 8, color: '#22c55e', symbol: 'diamond' }, name: 'Base',
                    showlegend: trajAnimFrame === 0
                },
                {
                    x: [xs[xs.length - 1]], y: [ys[ys.length - 1]], z: [zs[zs.length - 1]],
                    type: 'scatter3d', mode: 'markers',
                    marker: { size: 8, color: '#ef4444' }, name: 'End Effector',
                    showlegend: trajAnimFrame === 0
                },
                {
                    x: eeTrace.map(p => p[0]), y: eeTrace.map(p => p[1]), z: eeTrace.map(p => p[2]),
                    type: 'scatter3d', mode: 'lines',
                    line: { color: '#f97316', width: 2 }, name: 'Trace',
                    showlegend: trajAnimFrame === 0
                },
            ];

            Plotly.react('trajPlot3D', traces, {
                ...plotlyDarkLayout,
                title: {
                    text: `${type} — frame ${trajAnimFrame + 1}/${frames.length} | err: ${frame.position_error_mm.toFixed(2)} mm`,
                    font: { color: '#94a3b8', size: 12 }
                },
                showlegend: true,
                legend: { font: { color: '#94a3b8' }, bgcolor: 'rgba(0,0,0,0)' },
            });

            trajAnimFrame++;
        }, 120);

        btn.disabled = false;
        btn.textContent = '▶ Run Trajectory';
    } catch (e) {
        statsDiv.innerHTML = `<p style="color:#ef4444">Failed: ${e.message}</p>`;
        btn.disabled = false;
        btn.textContent = '▶ Run Trajectory';
    }
}
