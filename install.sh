#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}==>${NC} $1"; }
warn()  { echo -e "${YELLOW}Warning:${NC} $1"; }
error() { echo -e "${RED}Error:${NC} $1"; exit 1; }

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

# ─────────────────────────────────────────────────────────────────────────────
# macOS
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "macos" ]]; then

    # ── Homebrew ──────────────────────────────────────────────────────────────
    if ! command -v brew &>/dev/null; then
        step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        [[ -f /opt/homebrew/bin/brew ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        step "Homebrew already installed, skipping."
    fi

    # ── MySQL ─────────────────────────────────────────────────────────────────
    step "Installing MySQL..."
    if brew list mysql &>/dev/null; then
        warn "MySQL already installed, skipping brew install."
    else
        brew install mysql
    fi

    step "Starting MySQL service..."
    brew services start mysql
    for i in {1..10}; do
        mysqladmin ping -u root --silent 2>/dev/null && break
        sleep 1
    done

    step "Setting MySQL root password to 'my_password'..."
    if mysql -u root --connect-expired-password \
        -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my_password'; FLUSH PRIVILEGES;" \
        2>/dev/null; then
        echo "Password set."
    elif mysql -u root -pmy_password -e "SELECT 1;" &>/dev/null; then
        warn "MySQL root password is already 'my_password', nothing to do."
    else
        warn "Could not set password automatically. Run this manually:"
        echo "  mysql -u root -e \"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my_password'; FLUSH PRIVILEGES;\""
    fi

    # ── LaTeX (~200 MB) ───────────────────────────────────────────────────────
    step "Installing BasicTeX + required LaTeX packages..."
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

    # ── Python dependencies ───────────────────────────────────────────────────
    step "Installing Python dependencies..."
    if command -v pip3 &>/dev/null; then
        pip3 install -r requirements.txt
    else
        pip install -r requirements.txt
    fi

    echo -e "\n${GREEN}All done!${NC} Run: python main.py"
    echo ""
    echo "Note: if 'latex' is not found in a new terminal, add this to your shell profile:"
    echo "  export PATH=\"/Library/TeX/texbin:\$PATH\""

fi

# ─────────────────────────────────────────────────────────────────────────────
# Linux
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "linux" ]]; then

    step "Updating apt..."
    sudo apt-get update -qq

    # ── MySQL ─────────────────────────────────────────────────────────────────
    step "Installing MySQL..."
    sudo apt-get install -y mysql-server

    step "Starting MySQL service..."
    sudo systemctl start mysql
    for i in {1..10}; do
        sudo mysqladmin ping --silent 2>/dev/null && break
        sleep 1
    done

    step "Setting MySQL root password to 'my_password'..."
    # On Linux apt installs, root uses auth_socket — connect with sudo
    if sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my_password'; FLUSH PRIVILEGES;" \
        2>/dev/null; then
        echo "Password set."
    elif mysql -u root -pmy_password -e "SELECT 1;" &>/dev/null; then
        warn "MySQL root password is already 'my_password', nothing to do."
    else
        warn "Could not set password automatically. Run this manually:"
        echo "  sudo mysql -e \"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'my_password'; FLUSH PRIVILEGES;\""
    fi

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    step "Installing LaTeX packages..."
    sudo apt-get install -y \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        dvipng \
        ghostscript

    # ── Python dependencies ───────────────────────────────────────────────────
    step "Installing Python dependencies..."
    if command -v pip3 &>/dev/null; then
        pip3 install -r requirements.txt
    else
        sudo apt-get install -y python3-pip
        pip3 install -r requirements.txt
    fi

    echo -e "\n${GREEN}All done!${NC} Run: python3 main.py"

fi