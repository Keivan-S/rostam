#!/bin/sh
# Install rostam-server from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/rostamlabs/rostam/main/install.sh | sh
#
# Environment:
#   ROSTAM_VERSION       version to install (default: the latest release)
#   ROSTAM_INSTALL_DIR   where to put the binary (default: ~/.local/bin)
#   ROSTAM_NO_SUDO       set to refuse escalation and fail instead
#
# The download is verified against the checksums.txt published with the same
# release, and the script aborts on a mismatch. That check is the reason this
# script is worth having at all: piping a URL into a shell is only defensible if
# the thing it fetches is verified, and doing it by hand is the step people skip.
#
# These are PURE-GO builds. Everything works except WASM stored procedures,
# which return "wasm: stored procedures require a cgo build". If you need them,
# use the container image, which always carries cgo. `go install` carries it
# only when the build has cgo on: a native build (cgo defaults off when
# cross-compiling) on a machine with a C compiler.
#
# POSIX sh on purpose: this runs on whatever /bin/sh a machine happens to have.
set -eu

REPO="rostamlabs/rostam"
BINARY="rostam-server"
INSTALL_DIR="${ROSTAM_INSTALL_DIR:-$HOME/.local/bin}"

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf 'install: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

# ---- what are we running on ------------------------------------------------

detect_platform() {
	os=$(uname -s | tr '[:upper:]' '[:lower:]')
	arch=$(uname -m)
	case "$arch" in
	x86_64 | amd64) arch=amd64 ;;
	aarch64 | arm64) arch=arm64 ;;
	*) die "unsupported architecture: $arch. Build from source: go install github.com/$REPO/cmd/$BINARY@latest" ;;
	esac
	case "$os" in
	linux | darwin | freebsd) ;;
	*) die "unsupported OS: $os. Build from source: go install github.com/$REPO/cmd/$BINARY@latest" ;;
	esac
	# Combinations the release does not build. Naming them here gives a clear
	# message instead of a 404 from the download.
	case "$os/$arch" in
	freebsd/arm64) die "no freebsd/arm64 build is published. Build from source: go install github.com/$REPO/cmd/$BINARY@latest" ;;
	esac
	PLATFORM="${os}_${arch}"
}

# ---- which version ---------------------------------------------------------

latest_version() {
	# The redirect from /releases/latest carries the tag, which avoids the
	# GitHub API and its unauthenticated rate limit.
	url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") ||
		die "could not reach GitHub to resolve the latest release"
	tag=${url##*/}
	[ -n "$tag" ] && [ "$tag" != "latest" ] || die "could not determine the latest version; set ROSTAM_VERSION"
	printf '%s' "$tag"
}

# ---- checksum --------------------------------------------------------------

verify_checksum() {
	file=$1
	sums=$2
	want=$(awk -v f="$(basename "$file")" '$2 == f || $2 == "*"f {print $1}' "$sums" | head -n1)
	[ -n "$want" ] || die "no checksum published for $(basename "$file") — refusing to install"

	if command -v sha256sum >/dev/null 2>&1; then
		got=$(sha256sum "$file" | cut -d' ' -f1)
	elif command -v shasum >/dev/null 2>&1; then
		got=$(shasum -a 256 "$file" | cut -d' ' -f1)
	else
		die "neither sha256sum nor shasum is available — cannot verify the download"
	fi

	[ "$got" = "$want" ] || die "checksum mismatch for $(basename "$file")
  expected $want
  actual   $got
The download does not match what the release published. Not installing."
	printf '%s' "$got"
}

# ---- privileges ------------------------------------------------------------

# choose_writer decides how to write into INSTALL_DIR and sets SUDO to "" or
# "sudo". The default (~/.local/bin) never needs escalation; a system directory
# such as /usr/local/bin does, and failing with "permission denied" there is
# unhelpful when the user deliberately chose it.
#
# Escalation is announced before it happens and can be refused with
# ROSTAM_NO_SUDO. It works under `curl | sh` because sudo reads its password
# prompt from /dev/tty rather than stdin — stdin is the script itself here.
choose_writer() {
	SUDO=""
	if [ -d "$INSTALL_DIR" ]; then
		[ -w "$INSTALL_DIR" ] && return
	else
		parent=$(dirname "$INSTALL_DIR")
		[ -d "$parent" ] && [ -w "$parent" ] && return
	fi

	if [ -n "${ROSTAM_NO_SUDO:-}" ]; then
		die "$INSTALL_DIR is not writable and ROSTAM_NO_SUDO is set.
Choose a writable location instead:
  ROSTAM_INSTALL_DIR=\$HOME/.local/bin"
	fi
	command -v sudo >/dev/null 2>&1 || die "$INSTALL_DIR is not writable and sudo is not available.
Choose a writable location instead:
  ROSTAM_INSTALL_DIR=\$HOME/.local/bin"

	SUDO="sudo"
	warn "$INSTALL_DIR is not writable; using sudo to install there.
  (set ROSTAM_NO_SUDO to refuse, or ROSTAM_INSTALL_DIR to install without root)"
}

# ---- go --------------------------------------------------------------------

main() {
	need curl
	need tar
	need awk
	need mktemp
	detect_platform

	version=${ROSTAM_VERSION:-$(latest_version)}
	case "$version" in v*) ;; *) version="v$version" ;; esac

	# Assets are named without the leading v: rostam-server_0.1.1_linux_amd64.
	num=${version#v}
	# Every platform this script admits ships a tarball; the release's Windows
	# zip is unreachable here because detect_platform rejects Windows.
	asset="${BINARY}_${num}_${PLATFORM}.tar.gz"
	base="https://github.com/$REPO/releases/download/$version"

	# BSD mktemp (macOS, FreeBSD) rejects a bare -d and wants a template; GNU
	# mktemp accepts either. Both platforms are supported by detect_platform, so
	# the bare form would have failed on two of the three.
	tmp=$(mktemp -d 2>/dev/null || mktemp -d -t rostam-install) ||
		die "could not create a temporary directory"
	# shellcheck disable=SC2064 # expand tmp now, not at trap time
	trap "rm -rf '$tmp'" EXIT INT TERM

	log "Installing $BINARY $version ($PLATFORM)"

	curl -fsL -o "$tmp/$asset" "$base/$asset" 2>/dev/null ||
		die "could not download $asset
Either $version publishes no $PLATFORM build, or the download failed.
See https://github.com/$REPO/releases/tag/$version"
	curl -fsSL -o "$tmp/checksums.txt" "$base/checksums.txt" ||
		die "could not download checksums.txt — refusing to install unverified"

	digest=$(verify_checksum "$tmp/$asset" "$tmp/checksums.txt")
	log "  checksum ok: ${digest}"

	tar -xzf "$tmp/$asset" -C "$tmp" "$BINARY" || die "could not extract $BINARY from $asset"

	choose_writer
	# shellcheck disable=SC2086 # $SUDO is deliberately unquoted: empty means "no escalation"
	$SUDO mkdir -p "$INSTALL_DIR" || die "could not create $INSTALL_DIR"
	# Replace by rename so a running server keeps its open file rather than
	# having its executable rewritten underneath it.
	# shellcheck disable=SC2086
	$SUDO mv "$tmp/$BINARY" "$INSTALL_DIR/$BINARY.new" ||
		die "could not write to $INSTALL_DIR — set ROSTAM_INSTALL_DIR to a writable path"
	# shellcheck disable=SC2086
	$SUDO mv "$INSTALL_DIR/$BINARY.new" "$INSTALL_DIR/$BINARY" || die "could not replace $INSTALL_DIR/$BINARY"
	# shellcheck disable=SC2086
	$SUDO chmod +x "$INSTALL_DIR/$BINARY"

	# Confirm rather than announce: every write above can fail in ways that do
	# not stop the script (a sudo rule that permits nothing, a full disk), and
	# printing "installed" for a file that is not there is worse than failing.
	[ -x "$INSTALL_DIR/$BINARY" ] || die "$INSTALL_DIR/$BINARY is missing or not executable after install"

	log "  installed: $INSTALL_DIR/$BINARY"
	log ""
	"$INSTALL_DIR/$BINARY" -version 2>/dev/null || true

	case ":$PATH:" in
	*":$INSTALL_DIR:"*) ;;
	*) warn "
NOTE: $INSTALL_DIR is not on your PATH. Add it:
  export PATH=\"\$PATH:$INSTALL_DIR\"" ;;
	esac

	log "
Next:
  $BINARY -http 127.0.0.1:8080 -data ./data     # loopback needs no auth
  $BINARY -help-all                             # every flag
Docs: https://docs.rostamlabs.com/server/running/

These are pure-Go builds: WASM stored procedures need the container image, or
'go install github.com/$REPO/cmd/$BINARY@latest' built with cgo (native build,
C compiler installed)."
}

main "$@"
