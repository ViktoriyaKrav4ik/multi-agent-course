"""Перевірка Langfuse з .env (як homework-lesson-12/verify_langfuse.py)."""

from __future__ import annotations

from config import Settings, apply_langfuse_env_from_settings

apply_langfuse_env_from_settings()


def _print_auth_help() -> None:
    print()
    print("Langfuse 401 — перевірте ключі та LANGFUSE_BASE_URL (US vs EU).")


def main() -> None:
    s = Settings()
    if not s.langfuse_configured():
        print("Задайте LANGFUSE_PUBLIC_KEY та LANGFUSE_SECRET_KEY у .env")
        return
    print("BASE_URL:", s.langfuse_base_url)
    from langfuse import Langfuse

    client = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        base_url=s.langfuse_base_url,
    )
    try:
        ok = client.auth_check()
        print("auth_check():", ok)
        if not ok:
            _print_auth_help()
    except AttributeError:
        print("auth_check недоступний у цій версії SDK")
    except Exception as e:
        print("Помилка:", type(e).__name__, str(e).split("\n")[0][:200])
        _print_auth_help()


if __name__ == "__main__":
    main()
