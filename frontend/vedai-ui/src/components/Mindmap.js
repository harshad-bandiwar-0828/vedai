import React from "react";
import ReactFlow from "reactflow";
import "reactflow/dist/style.css";

const Mindmap = ({ text }) => {
  const lines = text.split("\n").filter(l => l.trim() !== "");

  const nodes = [];
  const edges = [];

  let id = 1;
  let parentStack = [];

  lines.forEach((line, index) => {
    const level = line.search(/\S/); // indentation level
    const label = line.trim();

    const nodeId = `${id++}`;

    nodes.push({
      id: nodeId,
      data: { label },
      position: { x: level * 60, y: index * 80 }
    });

    if (parentStack[level] !== undefined) {
      edges.push({
        id: `e-${parentStack[level]}-${nodeId}`,
        source: parentStack[level],
        target: nodeId
      });
    }

    parentStack[level] = nodeId;
  });

  return (
    <div style={{ width: "100%", height: "400px" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView />
    </div>
  );
};

export default Mindmap;