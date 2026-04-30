import { useEffect, useRef } from 'react';

/**
 * Dynamic particle-constellation background.
 * Nodes drift organically, connect with lines when close, and react to mouse.
 * Themed with teal colour scheme, adapts to dark/light mode.
 */

const DARK_RGB  = '45, 212, 191';   // #2DD4BF
const LIGHT_RGB = '15, 150, 131';   // #0f9683

const DENSITY          = 13000;
const MIN_NODES        = 55;
const MAX_NODES        = 160;
const BASE_SPEED       = 0.35;
const MAX_CONNECT_DIST = 190;
const MOUSE_RADIUS     = 160;
const MOUSE_STRENGTH   = 0.045;
const LINE_OPACITY_MAX = 0.22;
const NODE_OPACITY_MIN = 0.35;
const NODE_OPACITY_MAX = 0.70;
const ELECTRON_CHANCE  = 0.00075;
const MAX_ELECTRONS    = 30;

interface Node {
  x: number; y: number;
  vx: number; vy: number;
  r: number; op: number;
}

interface Electron {
  a: number; b: number;
  t: number; speed: number;
}

export default function BackgroundCanvas({ lightMode }: { lightMode: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<{
    nodes: Node[];
    electrons: Electron[];
    mouse: { x: number; y: number } | null;
    animId: number | null;
    W: number; H: number;
  }>({ nodes: [], electrons: [], mouse: null, animId: null, W: 0, H: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const state = stateRef.current;

    function rgb() { return lightMode ? LIGHT_RGB : DARK_RGB; }

    function createNodes() {
      const count = Math.max(MIN_NODES, Math.min(MAX_NODES, Math.floor((state.W * state.H) / DENSITY)));
      state.nodes = Array.from({ length: count }, () => {
        const angle = Math.random() * Math.PI * 2;
        const speed = BASE_SPEED * (0.5 + Math.random() * 0.8);
        return {
          x:  Math.random() * state.W,
          y:  Math.random() * state.H,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          r:  1.8 + Math.random() * 2.2,
          op: NODE_OPACITY_MIN + Math.random() * (NODE_OPACITY_MAX - NODE_OPACITY_MIN),
        };
      });
      state.electrons = [];
    }

    function resize() {
      state.W = canvas!.width  = window.innerWidth;
      state.H = canvas!.height = window.innerHeight;
      createNodes();
    }

    function updateNodes() {
      for (const n of state.nodes) {
        if (state.mouse) {
          const dx = state.mouse.x - n.x;
          const dy = state.mouse.y - n.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < MOUSE_RADIUS * MOUSE_RADIUS && d2 > 0) {
            const d = Math.sqrt(d2);
            const force = (1 - d / MOUSE_RADIUS) * MOUSE_STRENGTH;
            n.vx += (dx / d) * force;
            n.vy += (dy / d) * force;
          }
        }
        const spd = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
        const cap = BASE_SPEED * 2.8;
        if (spd > cap) { n.vx *= cap / spd; n.vy *= cap / spd; }
        n.vx *= 0.992;
        n.vy *= 0.992;
        if (spd < BASE_SPEED * 0.3) {
          n.vx += (Math.random() - 0.5) * 0.025;
          n.vy += (Math.random() - 0.5) * 0.025;
        }
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < -25) n.x = state.W + 25;
        else if (n.x > state.W + 25) n.x = -25;
        if (n.y < -25) n.y = state.H + 25;
        else if (n.y > state.H + 25) n.y = -25;
      }
    }

    function spawnElectrons() {
      if (state.electrons.length >= MAX_ELECTRONS) return;
      const len = state.nodes.length;
      for (let i = 0; i < len && state.electrons.length < MAX_ELECTRONS; i++) {
        for (let j = i + 1; j < len; j++) {
          if (Math.random() > ELECTRON_CHANCE) continue;
          const dx = state.nodes[j].x - state.nodes[i].x;
          const dy = state.nodes[j].y - state.nodes[i].y;
          if (dx * dx + dy * dy < MAX_CONNECT_DIST * MAX_CONNECT_DIST) {
            state.electrons.push({ a: i, b: j, t: 0, speed: 0.005 + Math.random() * 0.007 });
          }
        }
      }
    }

    function drawFrame() {
      ctx!.clearRect(0, 0, state.W, state.H);
      updateNodes();
      spawnElectrons();
      const color = rgb();
      const len = state.nodes.length;

      // Connection lines
      ctx!.lineWidth = 0.75;
      for (let i = 0; i < len; i++) {
        const a = state.nodes[i];
        for (let j = i + 1; j < len; j++) {
          const b = state.nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d2 = dx * dx + dy * dy;
          if (d2 >= MAX_CONNECT_DIST * MAX_CONNECT_DIST) continue;
          const op = LINE_OPACITY_MAX * (1 - Math.sqrt(d2) / MAX_CONNECT_DIST);
          ctx!.strokeStyle = `rgba(${color},${op.toFixed(3)})`;
          ctx!.beginPath();
          ctx!.moveTo(a.x, a.y);
          ctx!.lineTo(b.x, b.y);
          ctx!.stroke();
        }
      }

      // Nodes
      for (const n of state.nodes) {
        ctx!.beginPath();
        ctx!.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${color},${n.op.toFixed(3)})`;
        ctx!.fill();
      }

      // Electrons
      for (let i = state.electrons.length - 1; i >= 0; i--) {
        const e = state.electrons[i];
        e.t += e.speed;
        if (e.t > 1) { state.electrons.splice(i, 1); continue; }
        const a = state.nodes[e.a];
        const b = state.nodes[e.b];
        if (!a || !b) { state.electrons.splice(i, 1); continue; }
        const x = a.x + (b.x - a.x) * e.t;
        const y = a.y + (b.y - a.y) * e.t;
        ctx!.beginPath();
        ctx!.arc(x, y, 5, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${color},0.10)`;
        ctx!.fill();
        ctx!.beginPath();
        ctx!.arc(x, y, 2, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${color},0.90)`;
        ctx!.fill();
      }

      state.animId = requestAnimationFrame(drawFrame);
    }

    function pause()  { if (state.animId) { cancelAnimationFrame(state.animId); state.animId = null; } }
    function resume() { if (!state.animId) state.animId = requestAnimationFrame(drawFrame); }

    resize();
    state.animId = requestAnimationFrame(drawFrame);

    const onResize = () => resize();
    const onMouseMove = (e: MouseEvent) => { state.mouse = { x: e.clientX, y: e.clientY }; };
    const onMouseLeave = () => { state.mouse = null; };
    const onVisChange = () => {
      if (document.hidden) pause();
      else resume();
    };

    window.addEventListener('resize', onResize);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseleave', onMouseLeave);
    document.addEventListener('visibilitychange', onVisChange);

    return () => {
      pause();
      window.removeEventListener('resize', onResize);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseleave', onMouseLeave);
      document.removeEventListener('visibilitychange', onVisChange);
    };
  }, [lightMode]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  );
}
