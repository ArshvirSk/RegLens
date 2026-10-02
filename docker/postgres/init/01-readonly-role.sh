#!/bin/sh
# Runs once, on first initialisation of the postgres data volume.
#
# Creates the read-only role that the Phase 3 numeric/text-to-SQL tool will use.
# Least privilege is a security requirement (NFR "Security") and it is easier to
# enforce from day one than to retrofit: the application role can write, this role
# only ever SELECTs, and it cannot create objects or read the trace/metadata tables
# it does not need.
set -eu

RO_USER="${REGLENS_READONLY_USER:-reglens_ro}"
RO_PASSWORD="${REGLENS_READONLY_PASSWORD:-reglens_ro_local_dev}"
DB="${POSTGRES_DB:-reglens}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${RO_USER}') THEN
            CREATE ROLE ${RO_USER} LOGIN PASSWORD '${RO_PASSWORD}';
        END IF;
    END
    \$\$;

    GRANT CONNECT ON DATABASE ${DB} TO ${RO_USER};
    GRANT USAGE ON SCHEMA public TO ${RO_USER};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ${RO_USER};
    ALTER ROLE ${RO_USER} SET default_transaction_read_only = on;
    ALTER ROLE ${RO_USER} SET statement_timeout = '10s';
EOSQL

echo "[reglens] read-only role '${RO_USER}' ready"
