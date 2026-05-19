/* Neural IK Solver — Main JavaScript */

// ===== Global State =====
let solverMode = 'nn';
let heroScene, heroCamera, heroRenderer, heroArm;
let demoScene, demoCamera, demoRenderer, demoArm;

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    initHeroViz();
    initDemoViz();
    setupSliders();
    setupModeToggle();
    setupButtons();
    animateMetrics();
});

// ===== Robot Arm Visualization =====
// Renders robot arm from joint positions directly
class RobotArmViz {
    constructor(scene) {
        this.scene = scene;
        this.links = [];
        this.joints = [];
        this.endEffector = null;
        this.targetMarker = null;
        
        this.createMaterials();
        this.createBase();
        this.createTargetMarker();
    }
    
    createMaterials() {
        this.linkMaterial = new THREE.MeshStandardMaterial({
            color: 0xe8e6e3,
            metalness: 0.4,
            roughness: 0.6
        });
        this.jointMaterial = new THREE.MeshStandardMaterial({
            color: 0xd97706,
            metalness: 0.5,
            roughness: 0.4
        });
        this.eeMaterial = new THREE.MeshStandardMaterial({
            color: 0xd97706,
            metalness: 0.6,
            roughness: 0.3,
            emissive: 0xd97706,
            emissiveIntensity: 0.2
        });
        this.targetMaterial = new THREE.MeshBasicMaterial({
            color: 0x22c55e,
            transparent: true,
            opacity: 0.8
        });
    }
    
    createBase() {
        const baseGeo = new THREE.CylinderGeometry(0.15, 0.18, 0.05, 32);
        const baseMat = new THREE.MeshStandardMaterial({
            color: 0x2a2a2a,
            metalness: 0.3,
            roughness: 0.8
        });
        this.base = new THREE.Mesh(baseGeo, baseMat);
        this.base.position.y = 0.025;
        this.scene.add(this.base);
    }
    
    createTargetMarker() {
        const geo = new THREE.SphereGeometry(0.025, 16, 16);
        this.targetMarker = new THREE.Mesh(geo, this.targetMaterial);
        this.targetMarker.visible = false;
        this.scene.add(this.targetMarker);
        
        // Ring around target
        const ringGeo = new THREE.RingGeometry(0.03, 0.04, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0x22c55e,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5
        });
        this.targetRing = new THREE.Mesh(ringGeo, ringMat);
        this.targetRing.rotation.x = -Math.PI / 2;
        this.targetMarker.add(this.targetRing);
    }
    
    // Update arm from positions array [[x,y,z], ...]
    updateFromPositions(positions) {
        if (!positions || positions.length < 2) return;
        
        // Clear existing
        this.links.forEach(link => this.scene.remove(link));
        this.joints.forEach(joint => this.scene.remove(joint));
        if (this.endEffector) this.scene.remove(this.endEffector);
        this.links = [];
        this.joints = [];
        
        // Create joints and links
        for (let i = 0; i < positions.length; i++) {
            const pos = positions[i];
            // Convert: robot X->Three X, robot Y->Three -Z, robot Z->Three Y
            const x = pos[0];
            const y = pos[2];  // Z up becomes Y up
            const z = -pos[1]; // Y forward becomes -Z forward
            
            // Joint sphere
            const jointSize = i === 0 ? 0.05 : (i < positions.length - 1 ? 0.035 : 0.025);
            const jointGeo = new THREE.SphereGeometry(jointSize, 16, 16);
            const joint = new THREE.Mesh(jointGeo, this.jointMaterial);
            joint.position.set(x, y, z);
            this.scene.add(joint);
            this.joints.push(joint);
            
            // Link cylinder to previous joint
            if (i > 0) {
                const prevPos = positions[i - 1];
                const px = prevPos[0];
                const py = prevPos[2];
                const pz = -prevPos[1];
                
                const length = Math.sqrt(
                    Math.pow(x - px, 2) + 
                    Math.pow(y - py, 2) + 
                    Math.pow(z - pz, 2)
                );
                
                if (length > 0.01) {
                    const radius = 0.02 - i * 0.002;
                    const linkGeo = new THREE.CylinderGeometry(
                        Math.max(0.008, radius), 
                        Math.max(0.01, radius + 0.002), 
                        length, 
                        12
                    );
                    const link = new THREE.Mesh(linkGeo, this.linkMaterial);
                    
                    // Position at midpoint
                    link.position.set((x + px) / 2, (y + py) / 2, (z + pz) / 2);
                    
                    // Orient along direction
                    const dir = new THREE.Vector3(x - px, y - py, z - pz).normalize();
                    const up = new THREE.Vector3(0, 1, 0);
                    const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
                    link.setRotationFromQuaternion(quat);
                    
                    this.scene.add(link);
                    this.links.push(link);
                }
            }
        }
        
        // End effector cone
        const lastPos = positions[positions.length - 1];
        const eeGeo = new THREE.ConeGeometry(0.02, 0.05, 8);
        this.endEffector = new THREE.Mesh(eeGeo, this.eeMaterial);
        this.endEffector.position.set(lastPos[0], lastPos[2], -lastPos[1]);
        this.endEffector.rotation.x = Math.PI;
        this.scene.add(this.endEffector);
    }
    
    setTarget(x, y, z) {
        this.targetMarker.position.set(x, z, -y);
        this.targetMarker.visible = true;
    }
    
    hideTarget() {
        this.targetMarker.visible = false;
    }
    
    // Idle animation
    idleAnimation(time) {
        const t = time * 0.5;
        const positions = [
            [0, 0, 0],
            [0, 0, 0.67],
            [0.15 + 0.08 * Math.sin(t), 0.08 * Math.cos(t * 0.7), 0.32],
            [0.28 + 0.12 * Math.sin(t * 0.8), 0.12 * Math.cos(t * 0.6), 0.22],
            [0.35 + 0.15 * Math.sin(t * 0.9), 0.15 * Math.cos(t * 0.5), 0.45 + 0.08 * Math.sin(t * 1.1)],
            [0.35 + 0.15 * Math.sin(t * 0.9), 0.15 * Math.cos(t * 0.5), 0.45 + 0.08 * Math.sin(t * 1.1)],
            [0.35 + 0.15 * Math.sin(t * 0.9), 0.15 * Math.cos(t * 0.5), 0.45 + 0.08 * Math.sin(t * 1.1)]
        ];
        this.updateFromPositions(positions);
    }
}

// ===== Hero Visualization =====
function initHeroViz() {
    const container = document.getElementById('robotCanvas');
    if (!container) return;
    
    heroScene = new THREE.Scene();
    heroScene.background = new THREE.Color(0x1a1a1a);
    
    heroCamera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    heroCamera.position.set(1.5, 1.0, 1.5);
    heroCamera.lookAt(0, 0.35, 0);
    
    heroRenderer = new THREE.WebGLRenderer({ antialias: true });
    heroRenderer.setSize(container.clientWidth, container.clientHeight);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(heroRenderer.domElement);
    
    // Lights
    heroScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(5, 10, 5);
    heroScene.add(mainLight);
    const accentLight = new THREE.PointLight(0xd97706, 0.4, 10);
    accentLight.position.set(-2, 2, -2);
    heroScene.add(accentLight);
    
    // Grid
    heroScene.add(new THREE.GridHelper(2.5, 12, 0x333333, 0x252525));
    
    // Robot
    heroArm = new RobotArmViz(heroScene);
    
    // Animate
    let time = 0;
    function animate() {
        requestAnimationFrame(animate);
        time += 0.016;
        heroArm.idleAnimation(time);
        
        // Slow camera orbit
        const angle = time * 0.08;
        heroCamera.position.x = 1.5 * Math.cos(angle);
        heroCamera.position.z = 1.5 * Math.sin(angle);
        heroCamera.lookAt(0, 0.35, 0);
        
        heroRenderer.render(heroScene, heroCamera);
    }
    animate();
    
    window.addEventListener('resize', () => {
        heroCamera.aspect = container.clientWidth / container.clientHeight;
        heroCamera.updateProjectionMatrix();
        heroRenderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// ===== Demo Visualization =====
function initDemoViz() {
    const container = document.getElementById('demoCanvas');
    if (!container) return;
    
    demoScene = new THREE.Scene();
    demoScene.background = new THREE.Color(0x1a1a1a);
    
    demoCamera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    demoCamera.position.set(1.3, 0.9, 1.3);
    demoCamera.lookAt(0, 0.3, 0);
    
    demoRenderer = new THREE.WebGLRenderer({ antialias: true });
    demoRenderer.setSize(container.clientWidth, container.clientHeight);
    demoRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(demoRenderer.domElement);
    
    // Lights
    demoScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(5, 10, 5);
    demoScene.add(mainLight);
    
    // Grid
    demoScene.add(new THREE.GridHelper(2, 10, 0x333333, 0x252525));
    
    // Robot
    demoArm = new RobotArmViz(demoScene);
    
    // Initial pose
    demoArm.updateFromPositions([
        [0, 0, 0],
        [0, 0, 0.67],
        [0.2, 0, 0.35],
        [0.35, 0, 0.25],
        [0.4, 0, 0.5],
        [0.4, 0, 0.5],
        [0.4, 0, 0.5]
    ]);
    
    function animate() {
        requestAnimationFrame(animate);
        demoRenderer.render(demoScene, demoCamera);
    }
    animate();
    
    // Mouse drag
    let isDragging = false, prevX = 0, camAngle = Math.PI / 4;
    container.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; });
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', e => {
        if (!isDragging) return;
        camAngle += (e.clientX - prevX) * 0.01;
        demoCamera.position.x = 1.3 * Math.cos(camAngle);
        demoCamera.position.z = 1.3 * Math.sin(camAngle);
        demoCamera.lookAt(0, 0.3, 0);
        prevX = e.clientX;
    });
    
    window.addEventListener('resize', () => {
        demoCamera.aspect = container.clientWidth / container.clientHeight;
        demoCamera.updateProjectionMatrix();
        demoRenderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// ===== Sliders =====
function setupSliders() {
    const sliders = ['inputX', 'inputY', 'inputZ', 'inputRoll', 'inputPitch', 'inputYaw'];
    const vals = ['valX', 'valY', 'valZ', 'valRoll', 'valPitch', 'valYaw'];
    sliders.forEach((id, i) => {
        const slider = document.getElementById(id);
        const val = document.getElementById(vals[i]);
        if (slider && val) {
            slider.addEventListener('input', () => {
                val.textContent = parseFloat(slider.value).toFixed(2);
                updateTargetMarker();
            });
        }
    });
}

function updateTargetMarker() {
    if (!demoArm) return;
    const x = parseFloat(document.getElementById('inputX').value);
    const y = parseFloat(document.getElementById('inputY').value);
    const z = parseFloat(document.getElementById('inputZ').value);
    demoArm.setTarget(x, y, z);
}

// ===== Mode Toggle =====
function setupModeToggle() {
    const btns = document.querySelectorAll('.mode-btn');
    const note = document.getElementById('modeNote');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            solverMode = btn.dataset.mode;
            note.textContent = solverMode === 'nn' 
                ? 'Fast inference (~1ms), ~15-25mm error' 
                : 'High precision (~70ms), ~1mm error';
        });
    });
}

// ===== Buttons =====
function setupButtons() {
    document.getElementById('solveBtn')?.addEventListener('click', solveIK);
    document.getElementById('randomBtn')?.addEventListener('click', randomPose);
}

// ===== API =====
async function solveIK() {
    const x = document.getElementById('inputX').value;
    const y = document.getElementById('inputY').value;
    const z = document.getElementById('inputZ').value;
    const roll = document.getElementById('inputRoll').value;
    const pitch = document.getElementById('inputPitch').value;
    const yaw = document.getElementById('inputYaw').value;
    
    const resultDiv = document.getElementById('demoResult');
    resultDiv.innerHTML = '<p class="result-hint">Computing...</p>';
    
    try {
        const url = solverMode === 'nn'
            ? `/api/predict?x=${x}&y=${y}&z=${z}&roll=${roll}&pitch=${pitch}&yaw=${yaw}`
            : `/api/predict-fast?x=${x}&y=${y}&z=${z}&roll=${roll}&pitch=${pitch}&yaw=${yaw}`;
        
        const data = await (await fetch(url)).json();
        if (data.error) {
            resultDiv.innerHTML = `<p class="result-hint" style="color:#ef4444;">Error: ${data.error}</p>`;
            return;
        }
        
        displayResult(data);
        if (demoArm && data.arm_positions) {
            demoArm.updateFromPositions(data.arm_positions);
            demoArm.setTarget(parseFloat(x), parseFloat(y), parseFloat(z));
        }
    } catch (e) {
        resultDiv.innerHTML = `<p class="result-hint" style="color:#ef4444;">Failed: ${e.message}</p>`;
    }
}

async function randomPose() {
    const resultDiv = document.getElementById('demoResult');
    resultDiv.innerHTML = '<p class="result-hint">Generating...</p>';
    
    try {
        const url = solverMode === 'nn' ? '/api/random-demo' : '/api/random-demo-fast';
        const data = await (await fetch(url)).json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p class="result-hint" style="color:#ef4444;">Error: ${data.error}</p>`;
            return;
        }
        
        // Update sliders
        if (data.target_pose) {
            const ids = ['inputX', 'inputY', 'inputZ', 'inputRoll', 'inputPitch', 'inputYaw'];
            const vids = ['valX', 'valY', 'valZ', 'valRoll', 'valPitch', 'valYaw'];
            data.target_pose.forEach((v, i) => {
                document.getElementById(ids[i]).value = v;
                document.getElementById(vids[i]).textContent = v.toFixed(2);
            });
        }
        
        displayResult(data);
        if (demoArm && data.arm_positions) {
            demoArm.updateFromPositions(data.arm_positions);
            if (data.target_pose) demoArm.setTarget(data.target_pose[0], data.target_pose[1], data.target_pose[2]);
        }
    } catch (e) {
        resultDiv.innerHTML = `<p class="result-hint" style="color:#ef4444;">Failed: ${e.message}</p>`;
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById('demoResult');
    const posErr = data.position_error_mm;
    const posClass = posErr < 1 ? 'success' : (posErr < 5 ? 'warning' : '');
    
    let html = `
        <div class="result-grid">
            <div class="result-item">
                <span class="result-label">Position Error</span>
                <span class="result-value ${posClass}">${posErr.toFixed(3)} mm</span>
            </div>
            <div class="result-item">
                <span class="result-label">Orientation Error</span>
                <span class="result-value">${data.orientation_error_deg.toFixed(2)}°</span>
            </div>
            <div class="result-item">
                <span class="result-label">Inference Time</span>
                <span class="result-value">${data.inference_time_ms.toFixed(1)} ms</span>
            </div>
        </div>
        <div class="result-joints">
            <span class="result-joints-label">Joints (deg): </span>
            <span class="result-joints-value">[${data.joint_angles_deg.map(a => a.toFixed(1)).join(', ')}]</span>
        </div>
    `;
    
    if (data.nn_only) {
        const reduction = ((data.nn_only.position_error_mm - posErr) / data.nn_only.position_error_mm * 100).toFixed(0);
        html += `
            <div class="result-tto">
                <span class="result-joints-label">TTO: </span>
                <span class="result-joints-value">${data.nn_only.position_error_mm.toFixed(1)}mm → ${posErr.toFixed(2)}mm (${reduction}% better)</span>
            </div>
        `;
    }
    
    resultDiv.innerHTML = html;
}

// ===== Animate Metrics =====
function animateMetrics() {
    document.querySelectorAll('.metric-value[data-value]').forEach(el => {
        const target = parseFloat(el.dataset.value);
        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting) {
                animateNumber(el, target);
                observer.disconnect();
            }
        }, { threshold: 0.5 });
        observer.observe(el);
    });
}

function animateNumber(el, target) {
    const duration = 1200, start = performance.now();
    function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = target < 10 ? (target * eased).toFixed(2) : Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}
