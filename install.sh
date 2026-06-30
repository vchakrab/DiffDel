#!/bin/bash
#
# DiffDel setup script
# Designed to work on a clean macOS or Linux machine with nothing pre-installed.
# Safe to re-run — every step checks whether it's already done before acting.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}==>${NC} $1"; }
info()  { echo -e "  ${BLUE}-${NC} $1"; }
warn()  { echo -e "${YELLOW}Warning:${NC} $1"; }
error() { echo -e "${RED}Error:${NC} $1"; exit 1; }
have()  { command -v "$1" &>/dev/null; }

# ask "prompt text" "default value" -> echoes the chosen value
ask() {
    local prompt="$1" default="$2" reply
    if [[ -n "$default" ]]; then
        read -r -p "$prompt [$default]: " reply
    else
        read -r -p "$prompt: " reply
    fi
    echo "${reply:-$default}"
}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Detect platform ───────────────────────────────────────────────────────────
case "$(uname -s)" in
    Darwin)  PLATFORM="macos" ;;
    Linux)   PLATFORM="linux" ;;
    MINGW*|MSYS*|CYGWIN*)
        error "Windows detected. Please install WSL (Windows Subsystem for Linux) and re-run this script inside a WSL terminal." ;;
    *)
        error "Unsupported platform: $(uname -s)" ;;
esac

echo -e "\n${GREEN}Platform: ${PLATFORM}${NC}"

APT_UPDATED=""
apt_install() { sudo apt-get install -y "$@"; }
ensure_apt_updated() {
    if [[ -z "$APT_UPDATED" ]]; then
        step "Updating apt package index..."
        sudo apt-get update -qq
        APT_UPDATED=1
    fi
}

# ── MySQL root password setup (interactive, with sensible default) ──────────
setup_mysql_password() {
    local platform="$1"
    step "Configuring MySQL root password..."

    echo "DiffDel connects to MySQL using the root account."
    MYSQL_ROOT_PASSWORD="$(ask "Press Enter to use the default password, or type your own" "my_password")"

    if mysql -u root -p"${MYSQL_ROOT_PASSWORD}" -e "SELECT 1;" &>/dev/null; then
        warn "MySQL root password is already set to that value, nothing to do."
    else
        local alter_sql="ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '${MYSQL_ROOT_PASSWORD}'; FLUSH PRIVILEGES;"
        if [[ "$platform" == "linux" ]]; then
            # Fresh apt installs use auth_socket for root, so connect via sudo first.
            if sudo mysql -e "$alter_sql" 2>/dev/null; then
                echo "MySQL root password set."
            else
                warn "Could not set the password automatically. Run this manually:"
                echo "  sudo mysql -e \"$alter_sql\""
            fi
        else
            if mysql -u root --connect-expired-password -e "$alter_sql" 2>/dev/null; then
                echo "MySQL root password set."
            else
                warn "Could not set the password automatically. Run this manually:"
                echo "  mysql -u root -e \"$alter_sql\""
            fi
        fi
    fi

    echo "$MYSQL_ROOT_PASSWORD" > .mysql_root_password
    chmod 600 .mysql_root_password
    info "Saved to .mysql_root_password (keep this out of git — add it to .gitignore)."
}

# =============================================================================
# macOS setup
# =============================================================================
if [[ "$PLATFORM" == "macos" ]]; then

    if ! xcode-select -p &>/dev/null; then
        step "Installing Xcode Command Line Tools (needed for Homebrew and compiling)..."
        xcode-select --install || true
        error "A install popup should have appeared. Finish that install, then re-run this script."
    fi

    if ! have brew; then
        step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f /usr/local/bin/brew ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    else
        step "Homebrew already installed, skipping."
    fi

    if ! have git; then
        step "Installing git..."
        brew install git
    else
        step "git already installed, skipping."
    fi

    have curl || brew install curl

    if ! have python3; then
        step "Installing Python3..."
        brew install python3
    else
        step "Python3 already installed ($(python3 --version)), skipping."
    fi

    step "Installing MySQL..."
    if brew list mysql &>/dev/null; then
        warn "MySQL already installed, skipping brew install."
    else
        brew install mysql
    fi

    step "Starting MySQL service..."
    brew services start mysql
    for i in $(seq 1 30); do
        mysqladmin ping -u root --silent 2>/dev/null && break
        sleep 1
    done

    setup_mysql_password "macos"

    if have pdflatex; then
        step "LaTeX already installed, skipping."
    else
        step "Installing BasicTeX + required LaTeX packages (~200 MB)..."
        if brew list --cask basictex &>/dev/null; then
            warn "BasicTeX already installed, skipping."
        else
            brew install --cask basictex
        fi
        export PATH="/Library/TeX/texbin:$PATH"
        brew install ghostscript
        step "Installing LaTeX packages via tlmgr..."
        sudo tlmgr update --self
        sudo tlmgr install collection-latexrecommended dvipng
    fi

    echo ""
    echo "Note: if 'pdflatex' is not found in a new terminal, add this to your shell profile:"
    echo "  export PATH=\"/Library/TeX/texbin:\$PATH\""
fi

# =============================================================================
# Linux setup
# =============================================================================
if [[ "$PLATFORM" == "linux" ]]; then

    ensure_apt_updated

    if ! have git; then
        step "Installing git..."
        apt_install git
    else
        step "git already installed, skipping."
    fi

    if ! have curl; then
        step "Installing curl..."
        apt_install curl
    else
        step "curl already installed, skipping."
    fi

    step "Installing build tools (needed if any Python package compiles from source)..."
    apt_install build-essential python3-dev

    if ! have python3; then
        step "Installing Python3..."
        apt_install python3
    else
        step "Python3 already installed ($(python3 --version)), skipping."
    fi

    step "Ensuring python3-venv and pip are installed..."
    PY_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    apt_install "python${PY_MINOR}-venv" python3-pip || apt_install python3-venv python3-pip

    step "Installing MySQL..."
    if have mysql; then
        warn "MySQL client already installed, skipping."
    else
        apt_install mysql-server
    fi

    step "Starting MySQL service..."
    sudo systemctl enable mysql &>/dev/null || true
    sudo systemctl start mysql
    for i in $(seq 1 30); do
        sudo mysqladmin ping --silent 2>/dev/null && break
        sleep 1
    done

    setup_mysql_password "linux"

    if have pdflatex; then
        step "LaTeX already installed, skipping."
    else
        step "Installing LaTeX packages..."
        apt_install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended dvipng ghostscript
    fi
fi

# =============================================================================
# Python virtual environment + project dependencies
# =============================================================================
step "Setting up Python virtual environment..."
if [[ ! -d venv ]]; then
    python3 -m venv venv
else
    warn "venv/ already exists, reusing it."
fi

# shellcheck disable=SC1091
source venv/bin/activate

step "Upgrading pip..."
pip install --upgrade pip --quiet

if [[ -f requirements.txt ]]; then
    step "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    warn "No requirements.txt found, skipping."
fi

# =============================================================================
# Gurobi license setup
# =============================================================================
step "Gurobi license setup"

GUROBI_VERSION="$(python3 -c "import gurobipy as gp; print(gp.gurobi.version())" 2>/dev/null \
    | tr -d '() ' | tr ',' '.')" || GUROBI_VERSION=""

if [[ -z "$GUROBI_VERSION" ]]; then
    warn "Could not detect gurobipy — check that it installed correctly above."
else
    info "Detected gurobipy version: $GUROBI_VERSION"
fi

echo ""
echo "Gurobi needs a license to solve problems beyond the small built-in trial size."
echo "If you don't already have one:"
echo "  1. Create a free account at https://www.gurobi.com"
echo "     (use your university email for a free academic license)"
echo "  2. In the User Portal, go to Licenses and request/generate one"
echo "  3. Click the install icon next to it — it shows a command like:"
echo "       grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
echo "     Copy just the key (the UUID after grbgetkey)"
echo ""

GUROBI_KEY="$(ask "Paste your Gurobi license key now (or leave blank to skip and use the default restricted/trial license)" "")"

if [[ -n "$GUROBI_KEY" ]]; then
    if have grbgetkey; then
        GRBGETKEY_BIN="grbgetkey"
    else
        step "grbgetkey not found (pip-installed gurobipy doesn't include it) — downloading Gurobi license tools..."

        LT_VERSION="13.0.2"   # update this if the URL below 404s — check support.gurobi.com for the current version
        LT_DIR_VERSION="${LT_VERSION%.*}"

        if [[ "$PLATFORM" == "macos" ]]; then
            LT_FILE="licensetools${LT_VERSION}_macos_universal2.tar.gz"
        else
            LT_FILE="licensetools${LT_VERSION}_linux64.tar.gz"
        fi

        LT_URL="https://packages.gurobi.com/${LT_DIR_VERSION}/${LT_FILE}"
        TMP_DIR="$(mktemp -d)"
        GRBGETKEY_BIN=""

        if curl -fsSL -o "${TMP_DIR}/${LT_FILE}" "$LT_URL"; then
            tar -xzf "${TMP_DIR}/${LT_FILE}" -C "$TMP_DIR"
            GRBGETKEY_BIN="$(find "$TMP_DIR" -name grbgetkey -type f | head -n1)"
            [[ -n "$GRBGETKEY_BIN" ]] && chmod +x "$GRBGETKEY_BIN"
        else
            warn "Could not auto-download license tools from $LT_URL"
            warn "Visit https://support.gurobi.com and search for"
            warn "  'set up a license without installing the full package'"
            warn "download the tools for your platform manually, then run:"
            echo "  ./grbgetkey $GUROBI_KEY"
        fi
    fi

    if [[ -n "$GRBGETKEY_BIN" ]]; then
        step "Activating Gurobi license..."
        "$GRBGETKEY_BIN" "$GUROBI_KEY" || warn "grbgetkey did not complete successfully — you can re-run it manually later."
    fi
else
    info "Skipping Gurobi license activation. The bundled restricted/trial license will be used."
fi

# =============================================================================
# Final verification
# =============================================================================
step "Verifying installation..."

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}OK${NC}    $name"
    else
        echo -e "  ${RED}FAIL${NC}  $name"
    fi
}

check "git"      "have git"
check "python3"  "have python3"
check "pip"      "venv/bin/pip --version"
check "gurobipy" "venv/bin/python -c 'import gurobipy'"
check "mysql"    "have mysql"
check "pdflatex" "have pdflatex"

echo -e "\n${GREEN}All done!${NC} To get started:"
echo "  source venv/bin/activate"
echo "  python3 main.py"