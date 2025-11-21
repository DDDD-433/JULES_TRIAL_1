"""Bootstrap custom compatibility shims before loading Rasa actions."""

try:
    import sitecustomize  # type: ignore # noqa: F401
except Exception:
    # The shims are best-effort; the action server should still load.
    pass
