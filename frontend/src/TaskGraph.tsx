import { useMemo } from 'react';
import type { DagGraph, DagNode } from './api';

// ── Layout constants ──────────────────────────────────────────────────────────
const NODE_W = 200;
const NODE_H = 60;
const H_GAP = 32;   // horizontal gap between nodes in the same layer
const V_GAP = 80;   // vertical gap between layers
const PAD = 40;     // canvas padding

// ── Status colours ────────────────────────────────────────────────────────────
const STATUS_COLOR: Record<DagNode['status'], { fill: string; stroke: string; text: string }> = {
  pending:   { fill: '#1e293b', stroke: '#475569', text: '#94a3b8' },
  running:   { fill: '#422006', stroke: '#d97706', text: '#fbbf24' },
  completed: { fill: '#052e16', stroke: '#16a34a', text: '#4ade80' },
  failed:    { fill: '#2d0a0a', stroke: '#dc2626', text: '#f87171' },
  skipped:   { fill: '#1e1b4b', stroke: '#4f46e5', text: '#a5b4fc' },
  blocked:   { fill: '#431407', stroke: '#ea580c', text: '#fb923c' },
};

const STATUS_LABEL: Record<DagNode['status'], string> = {
  pending: 'Pending',
  running: 'Running…',
  completed: 'Done',
  failed: 'Failed',
  skipped: 'Skipped',
  blocked: 'Blocked',
};

// ── Layout computation ────────────────────────────────────────────────────────

interface LayoutNode extends DagNode {
  x: number;
  y: number;
  layer: number;
}

function computeLayout(graph: DagGraph): { nodes: LayoutNode[]; width: number; height: number } {
  if (graph.nodes.length === 0) return { nodes: [], width: 0, height: 0 };

  // Build dependency map: nodeId -> set of predecessor ids
  const deps: Map<string, Set<string>> = new Map();
  for (const node of graph.nodes) deps.set(node.id, new Set());
  for (const edge of graph.edges) deps.get(edge.to)?.add(edge.from);

  // Assign layer = longest path from any root
  const layer: Map<string, number> = new Map();
  const visited = new Set<string>();

  function getLayer(id: string): number {
    if (layer.has(id)) return layer.get(id)!;
    if (visited.has(id)) return 0; // cycle guard
    visited.add(id);
    const preds = deps.get(id) ?? new Set();
    const depth = preds.size === 0 ? 0 : Math.max(...[...preds].map(getLayer)) + 1;
    layer.set(id, depth);
    return depth;
  }

  for (const node of graph.nodes) getLayer(node.id);

  // Group by layer
  const byLayer: Map<number, DagNode[]> = new Map();
  for (const node of graph.nodes) {
    const l = layer.get(node.id) ?? 0;
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l)!.push(node);
  }

  const maxLayer = Math.max(...layer.values());
  const canvasWidth = Math.max(...[...byLayer.values()].map((nodes) =>
    nodes.length * NODE_W + (nodes.length - 1) * H_GAP
  )) + PAD * 2;

  const layoutNodes: LayoutNode[] = [];

  for (let l = 0; l <= maxLayer; l++) {
    const nodes = byLayer.get(l) ?? [];
    const rowWidth = nodes.length * NODE_W + (nodes.length - 1) * H_GAP;
    const startX = (canvasWidth - rowWidth) / 2;
    const y = PAD + l * (NODE_H + V_GAP);

    nodes.forEach((node, i) => {
      layoutNodes.push({
        ...node,
        layer: l,
        x: startX + i * (NODE_W + H_GAP),
        y,
      });
    });
  }

  const height = PAD + (maxLayer + 1) * (NODE_H + V_GAP) + PAD;
  return { nodes: layoutNodes, width: canvasWidth, height };
}

// ── Edge path (cubic bezier) ──────────────────────────────────────────────────

function edgePath(
  fromNode: LayoutNode,
  toNode: LayoutNode,
): string {
  const x1 = fromNode.x + NODE_W / 2;
  const y1 = fromNode.y + NODE_H;
  const x2 = toNode.x + NODE_W / 2;
  const y2 = toNode.y;
  const cy = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${cy}, ${x2} ${cy}, ${x2} ${y2}`;
}

// ── Main component ────────────────────────────────────────────────────────────

interface TaskGraphProps {
  graph: DagGraph;
}

export default function TaskGraph({ graph }: TaskGraphProps) {
  const { nodes, width, height } = useMemo(() => computeLayout(graph), [graph]);

  const nodeById = useMemo(() => {
    const m: Map<string, LayoutNode> = new Map();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);

  if (nodes.length === 0) {
    return (
      <div className="dag-empty">
        <span style={{ fontSize: 48, opacity: 0.3 }}>&#x25CB;</span>
        <p>No tasks yet. Send a message to see the task graph.</p>
      </div>
    );
  }

  return (
    <svg
      className="dag-svg"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
    >
      <defs>
        <marker
          id="arrowhead"
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
        >
          <path d="M 0 0 L 6 3 L 0 6 Z" fill="#475569" />
        </marker>
      </defs>

      {/* Edges */}
      {graph.edges.map((edge, i) => {
        const from = nodeById.get(edge.from);
        const to = nodeById.get(edge.to);
        if (!from || !to) return null;
        return (
          <path
            key={i}
            d={edgePath(from, to)}
            fill="none"
            stroke="#334155"
            strokeWidth={1.5}
            markerEnd="url(#arrowhead)"
          />
        );
      })}

      {/* Nodes */}
      {nodes.map((node) => {
        const colors = STATUS_COLOR[node.status] ?? STATUS_COLOR.pending;
        const label = node.description.length > 36
          ? node.description.slice(0, 33) + '…'
          : node.description;
        const roleLabel = node.role.charAt(0).toUpperCase() + node.role.slice(1);

        return (
          <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
            <rect
              width={NODE_W}
              height={NODE_H}
              rx={8}
              ry={8}
              fill={colors.fill}
              stroke={colors.stroke}
              strokeWidth={1.5}
            />
            {/* Status dot */}
            <circle cx={14} cy={NODE_H / 2} r={5} fill={colors.stroke} />
            {/* Task description */}
            <text
              x={26}
              y={NODE_H / 2 - 6}
              fill={colors.text}
              fontSize={11}
              fontWeight={600}
              fontFamily="inherit"
            >
              {label}
            </text>
            {/* Role + status pill */}
            <text
              x={26}
              y={NODE_H / 2 + 9}
              fill={colors.stroke}
              fontSize={10}
              fontFamily="inherit"
              opacity={0.85}
            >
              {roleLabel} · {STATUS_LABEL[node.status]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
