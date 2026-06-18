#!/bin/bash
set -euo pipefail

# AI Web Records — Local Development Setup for macOS (Apple Silicon)
# Run once to set up PostgreSQL, Redis, and the database.
# Prerequisites: Homebrew installed (https://brew.sh)

echo "=== AI Web Records Local Setup ==="

# 1. Install services if not present
echo "Checking dependencies..."

if ! command -v psql &> /dev/null; then
    echo "Installing PostgreSQL 16..."
    brew install postgresql@16
else
    echo "PostgreSQL already installed: $(psql --version)"
fi

if ! command -v redis-cli &> /dev/null; then
    echo "Installing Redis..."
    brew install redis
else
    echo "Redis already installed: $(redis-cli --version)"
fi

if ! command -v python3.12 &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Installing Python 3.12..."
    brew install python@3.12
else
    echo "Python already installed: $(python3 --version)"
fi

if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    brew install node
else
    echo "Node.js already installed: $(node --version)"
fi

# 2. Start services
echo ""
echo "Starting PostgreSQL..."
brew services start postgresql@16 2>/dev/null || echo "PostgreSQL already running"

echo "Starting Redis..."
brew services start redis 2>/dev/null || echo "Redis already running"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
for i in {1..10}; do
    if pg_isready -q 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! pg_isready -q 2>/dev/null; then
    echo "ERROR: PostgreSQL did not start. Check: brew services list"
    exit 1
fi

# 3. Create database
echo ""
echo "Setting up database..."
if psql -lqt | cut -d \| -f 1 | grep -qw medtimeline; then
    echo "Database 'medtimeline' already exists"
else
    createdb medtimeline
    echo "Created database 'medtimeline'"
fi

# 4. Install extensions
echo "Installing PostgreSQL extensions..."
psql medtimeline < "$(dirname "$0")/init-db.sql"

# 5. Apply performance tuning
echo "Applying PostgreSQL tuning for large imports..."
psql medtimeline < "$(dirname "$0")/pg-tuning.sql"

# 6. Restart PostgreSQL to apply shared_buffers change
echo "Restarting PostgreSQL to apply memory settings..."
brew services restart postgresql@16
sleep 3

# Wait for restart
for i in {1..10}; do
    if pg_isready -q 2>/dev/null; then
        break
    fi
    sleep 1
done

# 7. Configure Redis memory limit
echo ""
echo "Configuring Redis..."
REDIS_CONF="/opt/homebrew/etc/redis.conf"
if [ -f "$REDIS_CONF" ]; then
    if ! grep -q "^maxmemory 256mb" "$REDIS_CONF"; then
        echo "maxmemory 256mb" >> "$REDIS_CONF"
        echo "maxmemory-policy allkeys-lru" >> "$REDIS_CONF"
        brew services restart redis
        echo "Redis configured: maxmemory=256mb"
    else
        echo "Redis already configured"
    fi
else
    echo "WARNING: Redis config not found at $REDIS_CONF. Configure manually."
fi

# 8. Create .env from template if it doesn't exist
echo ""
if [ ! -f ../.env ] && [ -f ../.env.example ]; then
    cp ../.env.example ../.env
    echo "Created .env from .env.example — review and update secrets"
elif [ -f ../.env ]; then
    echo ".env already exists"
fi

# 9. Verify everything is running
echo ""
echo "=== Verification ==="
echo -n "PostgreSQL: " && pg_isready
echo -n "Redis: " && redis-cli ping
echo -n "Database: " && psql medtimeline -c "SELECT 'medtimeline OK' AS status;" -t -A

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  cd backend && alembic upgrade head    # Run migrations"
echo "  cd backend && python -m spacy download en_core_web_md  # PHI NER model (name redaction)"
echo "  cd backend && python scripts/build_terminology_index.py --refresh-live  # live RxNorm med index (offline-safe; also auto-refreshes on startup)"
echo "  cd backend && uvicorn app.main:app --reload --port 8000"
echo "  cd frontend && npm run dev"
echo ""
echo "To stop services later:"
echo "  brew services stop postgresql@16"
echo "  brew services stop redis"
