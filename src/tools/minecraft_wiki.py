from typing import Any, Dict

import aiohttp

WIKI_API_URL = "https://minecraft.wiki/api.php"

MINECRAFT_WIKI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_minecraft_wiki",
        "description": (
            "Look up the Minecraft Wiki page for a specific topic and return its text content. "
            "Use this to verify any specific numeric, mechanical, or version-specific fact "
            "(e.g. ore generation depths, enchantment behavior, trade ratios, crafting recipes) "
            "before stating it, rather than relying on memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page_title": {
                    "type": "string",
                    "description": "The wiki page title to look up, e.g. 'Diamond Ore', 'Villager trading', "
                                   "'Fire Resistance', 'Redstone Torch'",
                }
            },
            "required": ["page_title"],
        },
    },
}


async def fetch_minecraft_wiki_page(page_title: str, max_chars: int = 2000) -> str:
    """Fetch a Minecraft Wiki page's plain-text extract via the MediaWiki API.

    Never raises: network/HTTP failures and missing pages are returned as a
    plain-text message instead, since the result is fed back to a model as a
    tool response and a failed lookup should be something it can reason about,
    not a pipeline crash.
    """
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "exlimit": "1",
        "redirects": "1",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(WIKI_API_URL, params=params,
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return f"Error: Minecraft Wiki lookup failed with HTTP status {resp.status}."
                data = await resp.json()
    except Exception as exc:
        return f"Error: Minecraft Wiki lookup failed: {exc}"

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            return f"No wiki page found for '{page_title}'."
        extract = page.get("extract", "")
        if not extract:
            return f"No content found for wiki page '{page_title}'."
        return extract[:max_chars]
    return f"No wiki page found for '{page_title}'."


async def execute_minecraft_wiki_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Tool-executor callback matching ``ChatClient.call_chat_with_tools``'s contract."""
    if name != "search_minecraft_wiki":
        return f"Error: unknown tool '{name}'."
    page_title = arguments.get("page_title")
    if not page_title:
        return "Error: the 'page_title' argument is required."
    return await fetch_minecraft_wiki_page(page_title)
