from __future__ import annotations

import os

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine

# The same variables compose.yaml passes to the database container. On the Mac
# they come from .env: uv run --env-file .env python -m industrial_predictive_maintenance.db
REQUIRED = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def database_url() -> URL:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"{' '.join(missing)} not set. Run with: uv run --env-file .env ...")
    # Built from parts instead of one string, so a password with special
    # characters can never break the URL.
    return URL.create(
        "postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("IPM_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("IPM_DB_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
    )


def make_engine() -> Engine:
    return create_engine(database_url())


def main() -> None:
    url = database_url()
    with make_engine().connect() as conn:
        version = conn.execute(text("select version()")).scalar_one()
    # hide_password keeps the password out of the terminal.
    print("connected to", url.render_as_string(hide_password=True))
    print(version)


if __name__ == "__main__":
    main()