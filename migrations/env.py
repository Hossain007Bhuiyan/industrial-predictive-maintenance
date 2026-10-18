from alembic import context
from industrial_predictive_maintenance.db import make_engine
from industrial_predictive_maintenance.models import Base

# Migrations always run against the live database. Offline SQL scripts are not used.
with make_engine().connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()