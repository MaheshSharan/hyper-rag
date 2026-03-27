"""
Test script for fast project summarization
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.project_summarizer import ProjectSummarizer


def test_summarize(folder_path: str):
    """Test project summarization"""
    print(f"\n{'='*60}")
    print(f"Testing Project Summarization")
    print(f"⚡ FAST MODE (no LLM analysis)")
    print(f"{'='*60}\n")
    
    folder = Path(folder_path).resolve()
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    print(f"📂 Analyzing: {folder}")
    print(f"⏳ Should be fast (~3 seconds)...\n")
    
    # Create summarizer
    summarizer = ProjectSummarizer(str(folder))
    
    # Generate summary
    summary = summarizer.summarize(max_files_to_analyze=30)
    
    # Display results
    print(f"{'='*60}")
    print(f"SUMMARY RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Project: {summary['project_name']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Key Files Analyzed: {len(summary['key_files'])}\n")
    
    # Tech Stack
    tech = summary['tech_stack']
    print("🛠️  TECH STACK")
    print("-" * 40)
    if tech['languages']:
        print(f"Languages: {', '.join(tech['languages'])}")
    if tech['frameworks']:
        print(f"Frameworks: {', '.join(tech['frameworks'])}")
    if tech['package_managers']:
        print(f"Package Managers: {', '.join(tech['package_managers'])}")
    if tech['build_tools']:
        print(f"Build Tools: {', '.join(tech['build_tools'])}")
    if tech['dependencies']:
        print(f"Dependencies: {', '.join(tech['dependencies'][:8])}")
    print()
    
    # Structure
    structure = summary['structure']
    print("📁 PROJECT STRUCTURE")
    print("-" * 40)
    if structure['top_level_folders']:
        print(f"Top Folders: {', '.join(structure['top_level_folders'][:10])}")
    if structure['file_types']:
        print("\nFile Types:")
        for ext, count in list(structure['file_types'].items())[:8]:
            print(f"  {ext}: {count} files")
    print()
    
    # Key Files
    print("🔑 KEY FILES (Top 10)")
    print("-" * 40)
    for i, file_info in enumerate(summary['key_files'][:10], 1):
        print(f"{i}. {file_info['path']}")
        print(f"   Type: {file_info['type']} | Size: {file_info['size']} bytes")
        
        if 'structure' in file_info:
            struct = file_info['structure']
            if struct.get('classes'):
                print(f"   Classes: {', '.join(struct['classes'][:3])}")
            if struct.get('functions'):
                print(f"   Functions: {', '.join(struct['functions'][:3])}")
        print()
    
    # README
    if summary['readme_content']:
        print("📖 README PREVIEW")
        print("-" * 40)
        print(summary['readme_content'][:500])
        if len(summary['readme_content']) > 500:
            print("...")
        print()
    
    # LLM Analysis
    if summary.get('llm_analysis'):
        print("🤖 LLM INTELLIGENT ANALYSIS")
        print("=" * 60)
        print(summary['llm_analysis'])
        print("=" * 60)
        print()
    
    print(f"{'='*60}")
    print("✅ Summarization Complete!")
    print(f"{'='*60}\n")
    
    # Show cache location
    cache_dir = project_root / "data" / "processed" / summary['project_name']
    cache_file = cache_dir / "_project_summary.json"
    if cache_file.exists():
        print(f"💾 Summary cached at: {cache_file}")
        print(f"    (Will be reused on next call)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_summarizer.py <folder_path>")
        print("\nExample:")
        print("  python scripts/test_summarizer.py D:/Projects/my-app")
        print("  python scripts/test_summarizer.py .")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    test_summarize(folder_path)
