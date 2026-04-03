import React, { useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MarkerType
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";
import * as htmlToImage from "html-to-image";

// ==============================
// PARSE
// ==============================
const parseMindmap = (text) => {
  const lines = text
    .split("\n")
    .filter(l => l.trim() !== "")
    .filter(l => !l.toLowerCase().includes("mindmap"));

  let nodes = [];
  let edges = [];
  let id = 1;
  const stack = [];

  lines.forEach((line) => {
    const level = line.search(/\S/);
    const clean = line.trim().replace("-", "").trim();

    const nodeId = `${id++}`;

    nodes.push({
      id: nodeId,
      data: { label: clean, level },
      position: { x: 0, y: 0 }
    });

    while (stack.length && stack[stack.length - 1].level >= level) {
      stack.pop();
    }

    if (stack.length > 0) {
      edges.push({
        id: `e${stack[stack.length - 1].id}-${nodeId}`,
        source: stack[stack.length - 1].id,
        target: nodeId,
        markerEnd: { type: MarkerType.ArrowClosed }
      });
    }

    stack.push({ id: nodeId, level });
  });

  return { nodes, edges };
};

// ==============================
// LAYOUT
// ==============================
const layoutGraph = (nodes, edges) => {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));

  g.setGraph({
    rankdir: "LR",
    nodesep: 90,
    ranksep: 140
  });

  nodes.forEach((node) => {
    g.setNode(node.id, { width: 180, height: 60 });
  });

  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - 90,
        y: pos.y - 30
      }
    };
  });
};

// ==============================
// 🎨 LEVEL COLORS
// ==============================
const getColorByLevel = (level) => {
  if (level === 0) return "#4f46e5"; // main
  if (level < 4) return "#6366f1";
  if (level < 8) return "#8b5cf6";
  return "#334155";
};

// ==============================
// STYLE
// ==============================
const styleNodes = (nodes) => {
  return nodes.map((node) => {
    return {
      ...node,
      style: {
        background: getColorByLevel(node.data.level),
        color: "white",
        borderRadius: 10,
        padding: 10,
        maxWidth: 180,
        fontSize: 12,
        border: "1px solid rgba(255,255,255,0.1)",
        cursor: "pointer"
      }
    };
  });
};

// ==============================
// COMPONENT
// ==============================
function MindmapGraph({ text }) {
  const containerRef = useRef(null);
  const [hiddenNodes, setHiddenNodes] = useState([]);

  const { nodes, edges } = useMemo(() => {
    let parsed = parseMindmap(text);
    let styled = styleNodes(parsed.nodes);
    let layouted = layoutGraph(styled, parsed.edges);

    return {
      nodes: layouted,
      edges: parsed.edges
    };
  }, [text]);

  // ==============================
  // 🔽 COLLAPSE / EXPAND
  // ==============================
  const toggleNode = (nodeId) => {
    if (hiddenNodes.includes(nodeId)) {
      setHiddenNodes(hiddenNodes.filter(n => n !== nodeId));
    } else {
      setHiddenNodes([...hiddenNodes, nodeId]);
    }
  };

  const visibleNodes = nodes.filter(n => !hiddenNodes.includes(n.id));
  const visibleEdges = edges.filter(
    e => !hiddenNodes.includes(e.source) && !hiddenNodes.includes(e.target)
  );

  // ==============================
  // 📸 DOWNLOAD
  // ==============================
  const downloadImage = () => {
    htmlToImage.toPng(containerRef.current).then((dataUrl) => {
      const link = document.createElement("a");
      link.download = "vedai-mindmap.png";
      link.href = dataUrl;
      link.click();
    });
  };

  // ==============================
  // 🎯 CLICK HANDLER
  // ==============================
  const onNodeClick = (_, node) => {
    toggleNode(node.id);
  };

  return (
    <div className="mindmap-ultra">
      
      {/* TOP BAR */}
      <div className="mindmap-toolbar">
        <span>🧠 Mindmap</span>

        <div className="toolbar-actions">
          <button onClick={downloadImage}>📥 Download</button>
        </div>
      </div>

      {/* GRAPH */}
      <div
        ref={containerRef}
        style={{
          height: "520px",
          background: "#020617",
          borderRadius: "12px"
        }}
      >
        <ReactFlow
          nodes={visibleNodes}
          edges={visibleEdges}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
        >
          <Controls />
          <Background color="#334155" gap={20} />
        </ReactFlow>
      </div>
    </div>
  );
}

export default MindmapGraph;