"""CLI entry point — interactive REPL for Sentinel."""

from __future__ import annotations

import asyncio
import sys

from sentinel.app import SentinelApp


BANNER = """
╔═══════════════════════════════════════════╗
║           Sentinel v0.1.0                ║
║   Model-agnostic multi-agent framework    ║
╚═══════════════════════════════════════════╝

Type your request, or:
  /skills  — list loaded skills
  /trace   — show reasoning trace
  /config  — show current config
  /quit    — exit

"""


async def async_main(config_path: str | None = None):
    app = SentinelApp(config_path)

    # Discover skills
    n_skills = app.discover_skills()

    # Connect to MCP servers (if configured)
    n_mcp = await app.connect_mcp_servers()

    print(BANNER)
    print(f"  Tools: {', '.join(app.tools.names())}")
    print(f"  Skills: {n_skills} loaded")
    if n_mcp:
        print(f"  MCP tools: {n_mcp} from {len(app.mcp_manager.clients)} server(s)")
    print(f"  Model (strong): {app.config.models.strong}")
    print(f"  Model (fast): {app.config.models.fast}")
    print()

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye.")
            break

        if user_input == "/skills":
            skills = app.skills.all_skills
            if not skills:
                print("  No skills loaded.")
            else:
                for name, skill in skills.items():
                    print(f"  {name} v{skill.version} — {skill.description}")
            print()
            continue

        if user_input == "/trace":
            for step in app.trace.steps[-20:]:
                icon = {"observation": "👁", "thought": "💭", "action": "⚡",
                        "result": "✅", "error": "❌"}.get(step.step_type, "•")
                print(f"  {icon} [{step.agent_id[:12]}] {step.content[:150]}")
            print()
            continue

        if user_input == "/config":
            print(app.config.model_dump_json(indent=2))
            print()
            continue

        # Process the request
        try:
            response = await app.run(user_input)
            print(f"\nagent > {response}\n")
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}\n")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(async_main(config_path))


if __name__ == "__main__":
    main()
