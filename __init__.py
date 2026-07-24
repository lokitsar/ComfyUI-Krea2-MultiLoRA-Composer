"""ComfyUI Krea2 Multi-LoRA Composer custom node package."""

# Pytest imports a hyphenated custom-node directory's root __init__.py without a
# package name. ComfyUI always imports it as a package, which is when node
# registration should occur.
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # pragma: no cover - collection compatibility only
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
