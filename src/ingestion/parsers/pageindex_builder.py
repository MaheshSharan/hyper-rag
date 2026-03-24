from src.core.schemas import PageIndexNode
from typing import Dict, List
import hashlib
import re

class PageIndexBuilder:
	"""Builds hierarchical tree for documents (Markdown, PDFs with headings)"""

	def build_tree(self, text: str, source: str) -> List[PageIndexNode]:
		nodes: List[PageIndexNode] = []
		node_map: Dict[str, PageIndexNode] = {}

		lines = text.split('\n')
		current_path = []
		node_counter = 0

		for line in lines:
			line = line.strip()
			if not line:
				continue

			# Detect headings (Markdown style)
			heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
			if heading_match:
				level = len(heading_match.group(1))
				title = heading_match.group(2).strip()

				node_id = f"node_{hashlib.md5((source + title).encode()).hexdigest()[:8]}"

				# Pop path to correct level
				while len(current_path) >= level:
					current_path.pop()

				parent_id = current_path[-1] if current_path else None

				node = PageIndexNode(
					node_id=node_id,
					title=title,
					content="",  # content filled later
					level=level,
					parent_id=parent_id,
					children=[]
				)

				nodes.append(node)
				node_map[node_id] = node

				if parent_id and parent_id in node_map:
					node_map[parent_id].children.append(node_id)

				current_path.append(node_id)
				node_counter += 1

			else:
				# Add content to last node
				if nodes:
					nodes[-1].content += line + "\n"

		# Clean empty content
		for node in nodes:
			node.content = node.content.strip()

		return nodes
