//! VNET: run each per-user `--server` in its own transient systemd unit.
//!
//! Upstream starts the user's `--server` as `sudo -E -u <user> rustdesk --server` and keeps the `sudo`
//! `Child`. Stopping it SIGKILLs sudo, which cannot relay SIGKILL, so the server's own children
//! (`--tray`, `--cm`, ...) outlive it (upstream's "to-do" above `force_stop_server`). On a greeter ->
//! user-session switch that leaks the greeter user's tray for the rest of the boot; the new user's
//! server then tries `pkill -f "rustdesk --tray"` and gets EPERM on it.
//!
//! A transient unit gives the server a cgroup of its own: stopping the unit (KillMode=control-group)
//! kills every process in it, whatever it forked or whether it called setsid, with no pid bookkeeping.
//! Everything goes over the system bus (the `dbus` crate the fork already uses): no process spawns.
//!
//! What `sudo` provided and is reproduced here (boostlap, 2026-09-29):
//! * environment: built from an explicit list only (the caller's desktop variables, XDG_RUNTIME_DIR, a
//!   fixed PATH, and the locale/audio variables copied from the service's own environment). No
//!   EnvironmentFiles: they would override Environment= (systemd.exec precedence), and on Ubuntu
//!   /etc/environment only sets PATH, which is fixed here on purpose.
//! * limits: pam_limits entries from /etc/security/limits.conf{,.d} that apply to the user (by name,
//!   `*`, or @group), plus the open-files limit the sudo-started server had (1048576). Everything else
//!   is systemd's default for system units, the same defaults rustdesk.service itself runs with.
//! * umask 022 (pam_umask); supplementary groups via `User=`; cwd `/`, stdin null, output to the journal.
//! * logs: the server's output is in the journal under its own unit, not rustdesk.service:
//!   `journalctl -u 'rustdesk-server-*'` (units are collected when they stop, the journal stays).
//! * `PAMName=` is deliberately NOT set: it would open a logind session and move the processes into a
//!   session scope, outside the unit's cgroup. sudo's PAM stack on Ubuntu 24.04
//!   (common-session-noninteractive) has no pam_systemd, so today's server has no session either.
//!   pam_himmelblau / pam_exec home-private (Entra) prepare a user's home at sign-in; the server only
//!   ever starts for a session LightDM has already opened (or for the greeter user), so they already ran.

use dbus::{
    arg::{RefArg, Variant},
    blocking::{Connection, Proxy},
    Path as DbusPath,
};
use hbb_common::{log, ResultType};
use std::{
    cell::{Cell, RefCell},
    path::Path,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
    time::{Duration, Instant},
};

const DEST: &str = "org.freedesktop.systemd1";
const OBJ: &str = "/org/freedesktop/systemd1";
const MANAGER: &str = "org.freedesktop.systemd1.Manager";
const UNIT_IFACE: &str = "org.freedesktop.systemd1.Unit";
const NO_SUCH_UNIT: &str = "org.freedesktop.systemd1.NoSuchUnit";
const UNKNOWN_OBJECT: &str = "org.freedesktop.DBus.Error.UnknownObject";
pub const UNIT_PREFIX: &str = "rustdesk-server-";
/// Per-call D-Bus timeouts. The service loop ticks every 500 ms; a state probe must stay well under.
const START_CALL: Duration = Duration::from_secs(2);
const STOP_CALL: Duration = Duration::from_secs(1);
const PROBE_CALL: Duration = Duration::from_millis(250);
/// TimeoutStopSec of the unit; systemd SIGKILLs the cgroup after it.
const STOP_TIMEOUT_USEC: u64 = 2_000_000;
/// Total budget of one `stop()`, every call included (StopUnit, polling, KillUnit, polling).
const STOP_DEADLINE: Duration = Duration::from_millis(4_000);
/// sudo's secure_path on Ubuntu, minus /snap/bin (VNET OS removes snapd). The xrandr shim
/// (/usr/local/sbin/xrandr, Boost desktop/display/xrandr-rustdesk) relies on /usr/local/sbin first.
const SERVER_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
/// Copied from the service's own environment (sudo -E passed all of it; these are what the server uses).
const INHERITED_ENV: &[&str] = &[
    "LANG",
    "LANGUAGE",
    "PULSE_LATENCY_MSEC",
    "PIPEWIRE_LATENCY",
    "RUST_LOG",
];
/// Open-files limit the sudo-started server had on boostlap (soft = hard = 1048576).
const SUDO_NOFILE: u64 = 1_048_576;
/// A unit that exits this soon after starting counts as a failed start; one that has run this long
/// is healthy and resets the count.
const FAST_FAIL_WINDOW: Duration = Duration::from_secs(10);
/// After this many failed starts in a row the caller falls back to sudo ("degraded") ... Any early
/// exit counts, including a server that crashes by itself (a unit's exec failure and an app crash look
/// the same once the unit is collected); that is why degrading is bounded in time and a healthy run
/// ends the streak.
const FAST_FAIL_LIMIT: u32 = 3;
/// ... and tries units again after this long.
const DEGRADED_RETRY: Duration = Duration::from_secs(600);
/// Startup: how long listing leftover units may keep failing (bus down) before starting anyway.
const LIST_GIVE_UP: Duration = Duration::from_secs(30);

static SEQ: AtomicU64 = AtomicU64::new(0);
static FAST_FAILS: AtomicU32 = AtomicU32::new(0);
static DEGRADED_AT: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);

/// The request never reached the bus (no connection): systemd cannot have acted on it.
#[derive(Debug)]
struct NotSent(dbus::Error);

impl std::fmt::Display for NotSent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "system bus unavailable: {}", self.0)
    }
}

impl std::error::Error for NotSent {}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// A D-Bus variant of any argument type (for the a(sv) property list).
fn v<T: RefArg + 'static>(x: T) -> Variant<Box<dyn RefArg>> {
    Variant(Box::new(x))
}

thread_local! {
    // The service loop is single-threaded; one system-bus connection is reused across ticks.
    static BUS: RefCell<Option<Connection>> = RefCell::new(None);
}

/// Errors that say nothing about the connection itself (the connection is kept).
fn is_benign(e: &dbus::Error) -> bool {
    !is_connection_error(e)
}

/// Errors after which the cached connection may be dead and is replaced on the next call.
fn is_connection_error(e: &dbus::Error) -> bool {
    match e.name() {
        None => true,
        Some(n) => matches!(
            n,
            "org.freedesktop.DBus.Error.Disconnected"
                | "org.freedesktop.DBus.Error.NoReply"
                | "org.freedesktop.DBus.Error.Timeout"
                | "org.freedesktop.DBus.Error.TimedOut"
                | "org.freedesktop.DBus.Error.NoServer"
                | "org.freedesktop.DBus.Error.IOError"
                | "org.freedesktop.DBus.Error.ServiceUnknown"
        ),
    }
}

/// Run `f` against a proxy for `path` with `timeout`. A connection error drops the cached
/// connection, so a dead one is replaced on the next call; a failed connect is `NotSent`.
/// (Connecting is a local unix-socket connect + auth; it has no timeout of its own.)
fn with_proxy<T>(
    path: DbusPath<'static>,
    timeout: Duration,
    f: impl FnOnce(&Proxy<&Connection>) -> Result<T, dbus::Error>,
) -> ResultType<T> {
    BUS.with(|bus| {
        let mut bus = bus.borrow_mut();
        if bus.is_none() {
            *bus = Some(Connection::new_system().map_err(NotSent)?);
        }
        let res = {
            let conn = bus.as_ref().unwrap();
            f(&conn.with_proxy(DEST, path, timeout))
        };
        if let Err(e) = &res {
            if !is_benign(e) {
                *bus = None;
            }
        }
        Ok(res?)
    })
}

fn with_manager<T>(
    timeout: Duration,
    f: impl FnOnce(&Proxy<&Connection>) -> Result<T, dbus::Error>,
) -> ResultType<T> {
    with_proxy(DbusPath::from(OBJ), timeout, f)
}

fn dbus_err_is<T>(res: &ResultType<T>, names: &[&str]) -> bool {
    match res {
        Err(e) => e
            .downcast_ref::<dbus::Error>()
            .and_then(|e| e.name())
            .map_or(false, |n| names.contains(&n)),
        Ok(_) => false,
    }
}

/// `name` must be one of ours: prefix, then [0-9-], then ".service". Guards every stop/kill.
pub fn is_our_unit_name(name: &str) -> bool {
    name.strip_prefix(UNIT_PREFIX)
        .and_then(|rest| rest.strip_suffix(".service"))
        .map_or(false, |mid| {
            !mid.is_empty() && mid.bytes().all(|b| b.is_ascii_digit() || b == b'-')
        })
}

/// A uid we will run a server as: a plain decimal in 1..=u32::MAX-1 (never root, never the
/// overflow uid -1).
pub fn parse_uid(uid: &str) -> Option<u32> {
    if uid.is_empty() || !uid.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    match uid.parse::<u32>() {
        Ok(0) | Ok(u32::MAX) | Err(_) => None,
        Ok(n) => Some(n),
    }
}

/// The unit the calling process runs in (e.g. "rustdesk.service"), from /proc/self/cgroup.
fn own_unit() -> Option<String> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    own_unit_from_cgroup(&cg)
}

fn own_unit_from_cgroup(cg: &str) -> Option<String> {
    let path = cg.lines().find_map(|l| l.strip_prefix("0::"))?;
    let last = path.rsplit('/').next()?;
    (last.ends_with(".service") && last.len() > ".service".len()).then(|| last.to_owned())
}

pub fn unit_name(uid: u32, service_pid: u32, seq: u64) -> String {
    format!("{UNIT_PREFIX}{uid}-{service_pid}-{seq}.service")
}

fn inherit_key(k: &str) -> bool {
    INHERITED_ENV.contains(&k) || k.starts_with("LC_")
}

/// The `Environment=` list: fixed XDG_RUNTIME_DIR and PATH first (nothing overrides them), then the
/// caller's variables, then the allow-listed part of `inherited` (the service's environment). Keys are
/// validated; values with a newline or NUL are dropped; the first occurrence of a key wins.
pub fn build_environment(
    uid: u32,
    envs: &[(String, String)],
    inherited: &[(String, String)],
) -> Vec<String> {
    fn valid_key(k: &str) -> bool {
        let mut it = k.bytes();
        matches!(it.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_')
            && it.all(|c| c.is_ascii_alphanumeric() || c == b'_')
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut push = |k: &str, v: &str| {
        if !valid_key(k) || v.contains('\n') || v.contains('\0') || seen.iter().any(|s| s == k) {
            return;
        }
        seen.push(k.to_owned());
        out.push(format!("{k}={v}"));
    };
    push("XDG_RUNTIME_DIR", &format!("/run/user/{uid}"));
    push("PATH", SERVER_PATH);
    for (k, v) in envs {
        push(k, v);
    }
    for (k, v) in inherited.iter().filter(|(k, _)| inherit_key(k)) {
        push(k, v);
    }
    out
}

/// pam_limits items we map, to their systemd property names.
const LIMIT_ITEMS: &[(&str, &str)] = &[
    ("nofile", "LimitNOFILE"),
    ("memlock", "LimitMEMLOCK"),
    ("rtprio", "LimitRTPRIO"),
    ("nice", "LimitNICE"),
    ("nproc", "LimitNPROC"),
    ("core", "LimitCORE"),
    ("stack", "LimitSTACK"),
    ("as", "LimitAS"),
    ("fsize", "LimitFSIZE"),
    ("data", "LimitDATA"),
    ("cpu", "LimitCPU"),
];

/// How specific a limits.conf domain is, pam_limits style: lower = more specific. A line only
/// overrides a side (soft/hard) that was set by an equally or less specific line, so `alice` beats a
/// later `*`, and among equals the later line wins (limits.conf, then limits.d in name order).
fn domain_priority(domain: &str, user: &str, uid: u32, groups: &[String]) -> Option<u8> {
    if domain == user {
        return Some(0);
    }
    // uid ranges: "min:max", "min:", ":max", or "n:" (pam_limits: a user-class match).
    if let Some((lo, hi)) = domain.split_once(':') {
        if !domain.starts_with('@') && !domain.starts_with('%') {
            let lo = if lo.is_empty() { Some(0) } else { lo.parse::<u32>().ok() };
            let hi = if hi.is_empty() { Some(u32::MAX) } else { hi.parse::<u32>().ok() };
            return match (lo, hi) {
                (Some(lo), Some(hi)) if lo <= uid && uid <= hi => Some(0),
                _ => None,
            };
        }
    }
    if let Some(g) = domain.strip_prefix('@') {
        return groups.iter().any(|x| x == g).then_some(1);
    }
    (domain == "*").then_some(3)
}

/// Effective (soft, hard) per systemd property, from pam_limits text `sources` for `user`/`uid` in
/// `groups`. Only the side(s) a line names are set (`soft`, `hard`, or both for `-`); a side no line
/// names stays None. Units are converted to what systemd expects: KiB -> bytes for
/// memlock/core/stack/as/fsize/data, minutes -> seconds for cpu, nice as the 20-based rlimit value
/// (clamped to -20..19; "-1"/"unlimited" for nice means nice -1, as in pam_limits). nofile
/// "unlimited" becomes `nr_open` (pam_limits does the same). Unknown items/values are ignored.
pub fn pam_limits_for(
    sources: &[String],
    user: &str,
    uid: u32,
    groups: &[String],
    nr_open: u64,
) -> Vec<(&'static str, Option<u64>, Option<u64>)> {
    // (prop, soft (value, prio), hard (value, prio))
    type Side = Option<(u64, u8)>;
    let mut out: Vec<(&'static str, Side, Side)> = Vec::new();
    for text in sources {
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 4 {
                continue;
            }
            let (domain, kind, item, value) = (f[0], f[1], f[2], f[3]);
            let Some(prio) = domain_priority(domain, user, uid, groups) else {
                continue;
            };
            let Some(&(_, prop)) = LIMIT_ITEMS.iter().find(|(i, _)| *i == item) else {
                continue;
            };
            let unlimited = matches!(value.to_ascii_lowercase().as_str(), "unlimited" | "infinity" | "-1");
            let val = if item == "nice" {
                let n = if unlimited {
                    -1
                } else {
                    match value.parse::<i64>() {
                        Ok(n) => n.clamp(-20, 19),
                        Err(_) => continue,
                    }
                };
                (20 - n) as u64
            } else if unlimited {
                if item == "nofile" {
                    nr_open
                } else {
                    u64::MAX
                }
            } else {
                let Ok(n) = value.parse::<u64>() else {
                    continue;
                };
                match item {
                    "memlock" | "core" | "stack" | "as" | "fsize" | "data" => {
                        n.saturating_mul(1024)
                    }
                    "cpu" => n.saturating_mul(60),
                    _ => n,
                }
            };
            let idx = match out.iter().position(|(p, _, _)| *p == prop) {
                Some(i) => i,
                None => {
                    out.push((prop, None, None));
                    out.len() - 1
                }
            };
            let set = |side: &mut Side| {
                if side.map_or(true, |(_, p)| prio <= p) {
                    *side = Some((val, prio));
                }
            };
            match kind {
                "soft" => set(&mut out[idx].1),
                "hard" => set(&mut out[idx].2),
                "-" => {
                    set(&mut out[idx].1);
                    set(&mut out[idx].2);
                }
                _ => {}
            }
        }
    }
    out.into_iter()
        .map(|(p, s, h)| (p, s.map(|x| x.0), h.map(|x| x.0)))
        .collect()
}

/// The systemd properties for a limit set: `LimitX` = hard, `LimitXSoft` = soft, only for the sides
/// that are set (an unset side keeps systemd's default for system units, as it did under sudo). A soft
/// value above an explicit hard one is clamped. NOFILE defaults to what the sudo-started server had.
pub fn limit_props_from(limits: Vec<(&'static str, Option<u64>, Option<u64>)>) -> Vec<(String, u64)> {
    let mut limits = limits;
    match limits.iter_mut().find(|(p, _, _)| *p == "LimitNOFILE") {
        Some(l) => {
            l.1 = l.1.or(Some(SUDO_NOFILE.min(l.2.unwrap_or(u64::MAX))));
            l.2 = l.2.or(Some(SUDO_NOFILE.max(l.1.unwrap_or(0))));
        }
        None => limits.push(("LimitNOFILE", Some(SUDO_NOFILE), Some(SUDO_NOFILE))),
    }
    let mut props = Vec::new();
    for (prop, soft, hard) in limits {
        if let Some(h) = hard {
            props.push((prop.to_owned(), h));
        }
        if let Some(s) = soft {
            props.push((format!("{prop}Soft"), s.min(hard.unwrap_or(u64::MAX))));
        }
    }
    props
}

fn read_limits_sources() -> Vec<String> {
    let mut sources = Vec::new();
    if let Ok(s) = std::fs::read_to_string("/etc/security/limits.conf") {
        sources.push(s);
    }
    if let Ok(dir) = std::fs::read_dir("/etc/security/limits.d") {
        let mut files: Vec<_> = dir
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "conf"))
            .collect();
        files.sort();
        for p in files {
            if let Ok(s) = std::fs::read_to_string(p) {
                sources.push(s);
            }
        }
    }
    sources
}

/// systemd `Limit*` properties for `uid`: pam_limits parity plus the sudo open-files limit.
fn limit_properties(uid: u32) -> Vec<(String, u64)> {
    use hbb_common::users::{get_user_by_uid, get_user_groups};
    let (name, groups) = match get_user_by_uid(uid) {
        Some(u) => {
            let name = u.name().to_string_lossy().into_owned();
            let groups = get_user_groups(&name, u.primary_group_id())
                .unwrap_or_default()
                .iter()
                .map(|g| g.name().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            (name, groups)
        }
        None => (String::new(), Vec::new()),
    };
    let nr_open = std::fs::read_to_string("/proc/sys/fs/nr_open")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(SUDO_NOFILE);
    limit_props_from(pam_limits_for(
        &read_limits_sources(),
        &name,
        uid,
        &groups,
        nr_open,
    ))
}

pub struct ServerUnit {
    name: String,
    started: Instant,
    stopping: Cell<bool>,
    /// Already counted toward (or cleared) the fast-fail streak.
    counted: Cell<bool>,
}

/// Why a start produced no unit.
pub enum StartError {
    /// systemd definitely did not create it: the caller may use another way to start a server.
    NotCreated(hbb_common::anyhow::Error),
    /// Too many units failed right after starting: the caller should fall back.
    Degraded,
}

impl ServerUnit {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// True while repeated units have died right after starting (the caller uses its fallback);
    /// after DEGRADED_RETRY units are tried again.
    pub fn degraded() -> bool {
        if FAST_FAILS.load(Ordering::Relaxed) < FAST_FAIL_LIMIT {
            return false;
        }
        let mut at = DEGRADED_AT.lock().unwrap_or_else(|e| e.into_inner());
        let since = *at.get_or_insert_with(Instant::now);
        if since.elapsed() >= DEGRADED_RETRY {
            log::warn!("server units: retrying after {:?} degraded", since.elapsed());
            FAST_FAILS.store(0, Ordering::Relaxed);
            *at = None;
            return false;
        }
        true
    }

    /// Start `exe args...` as `uid` in a new transient unit. Only an outcome where systemd provably
    /// did not create the unit is `NotCreated` (the request never left: no bus connection; or systemd
    /// answered with an error). A lost reply (timeout, disconnect) is ambiguous: systemd may still act
    /// on the request, so the unit is tracked by name and never paired with a fallback; if it never
    /// comes up, `has_exited()` says so and the next start is again a unit.
    pub fn start(
        uid: &str,
        exe: &Path,
        args: &[&str],
        envs: &[(String, String)],
    ) -> Result<ServerUnit, StartError> {
        if Self::degraded() {
            return Err(StartError::Degraded);
        }
        let not_created = |e: hbb_common::anyhow::Error| Err(StartError::NotCreated(e));
        let Some(uid) = parse_uid(uid) else {
            return not_created(hbb_common::anyhow::anyhow!(
                "refusing a server unit for uid '{uid}'"
            ));
        };
        let Some(exe_s) = exe.to_str() else {
            return not_created(hbb_common::anyhow::anyhow!("non-UTF-8 executable path"));
        };
        let name = unit_name(uid, std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
        let mut argv = vec![exe_s.to_owned()];
        argv.extend(args.iter().map(|a| a.to_string()));
        let inherited: Vec<(String, String)> = std::env::vars().collect();
        let environment = build_environment(uid, envs, &inherited);

        let mut props: Vec<(String, Variant<Box<dyn RefArg>>)> = vec![
            (
                "Description".into(),
                v(format!("RustDesk server for uid {uid}")),
            ),
            ("ExecStart".into(), v(vec![(exe_s.to_owned(), argv, false)])),
            ("User".into(), v(uid.to_string())),
            ("Environment".into(), v(environment)),
            ("UMask".into(), v(0o022u32)),
            ("KillMode".into(), v("control-group".to_owned())),
            ("TimeoutStopUSec".into(), v(STOP_TIMEOUT_USEC)),
            // Gone as soon as it stops or fails: no "failed" leftovers; names are never reused anyway.
            ("CollectMode".into(), v("inactive-or-failed".to_owned())),
        ];
        if systemd_version() >= 255 {
            // HOME, SHELL, LOGNAME from the passwd entry (sudo gave the server SHELL and
            // terminal_service.rs uses it). Environment= still wins for HOME when the desktop has one.
            // Older systemd rejects the property (the whole start would fail), so it is only sent here.
            props.push(("SetLoginEnvironment".into(), v(true)));
        }
        for (prop, val) in limit_properties(uid) {
            props.push((prop, v(val)));
        }
        if let Some(parent) = own_unit() {
            // Stopping rustdesk.service stops every server it started.
            props.push(("BindsTo".into(), v(vec![parent.clone()])));
            props.push(("After".into(), v(vec![parent])));
        }
        let aux: Vec<(String, Vec<(String, Variant<Box<dyn RefArg>>)>)> = vec![];
        let res = with_manager(START_CALL, |m| {
            let (_job,): (DbusPath,) = m.method_call(
                MANAGER,
                "StartTransientUnit",
                (name.as_str(), "fail", props, aux),
            )?;
            Ok(())
        });
        let unit = ServerUnit {
            name,
            started: Instant::now(),
            stopping: Cell::new(false),
            counted: Cell::new(false),
        };
        match res {
            Ok(()) => {
                log::info!("started {} (uid {uid})", unit.name);
                Ok(unit)
            }
            Err(e) if start_error_is_definite(&e) => not_created(e),
            Err(e) => {
                log::warn!("{}: start outcome unknown ({e}); tracking it", unit.name);
                Ok(unit)
            }
        }
    }

    /// True once the server has exited (unit inactive, failed or gone). A bus error counts as
    /// "still running": restarting on a transient bus hiccup would start a second server. Counts
    /// exits within FAST_FAIL_WINDOW of the start (that we did not ask for) toward `degraded()`.
    pub fn has_exited(&self) -> bool {
        let exited = match active_state(&self.name, PROBE_CALL) {
            Ok(None) => true,
            Ok(Some(s)) => s == "inactive" || s == "failed",
            Err(e) => {
                log::warn!("{}: cannot read state: {e}", self.name);
                false
            }
        };
        let age = self.started.elapsed();
        if !self.counted.get() && !self.stopping.get() {
            if exited && age < FAST_FAIL_WINDOW {
                self.counted.set(true);
                let n = FAST_FAILS.fetch_add(1, Ordering::Relaxed) + 1;
                log::error!("{} exited {age:?} after starting ({n} in a row)", self.name);
            } else if age >= FAST_FAIL_WINDOW {
                // Ran long enough (whether it is still up or exited later): the streak is over.
                self.counted.set(true);
                FAST_FAILS.store(0, Ordering::Relaxed);
            }
        }
        exited
    }

    /// Stop the unit; true once every process in it is gone. False = still running (or unknown):
    /// the caller keeps owning it and retries.
    pub fn stop(&self) -> bool {
        self.stopping.set(true);
        stop_unit_by_name(&self.name)
    }
}

/// systemd's major version (Manager.Version, e.g. "255.4-1ubuntu8.17" -> 255), read once; 0 if unknown.
fn systemd_version() -> u32 {
    static VERSION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        with_manager(PROBE_CALL, |m| {
            use dbus::blocking::stdintf::org_freedesktop_dbus::Properties;
            m.get::<String>(MANAGER, "Version")
        })
        .ok()
        .and_then(|v| parse_systemd_version(&v))
        .unwrap_or(0)
    })
}

fn parse_systemd_version(v: &str) -> Option<u32> {
    let digits: String = v
        .trim_start_matches(|c: char| !c.is_ascii_digit())
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// A StartTransientUnit failure that proves systemd did not create the unit: the request never left
/// (no connection), or systemd itself answered with an error. Lost replies are not definite.
fn start_error_is_definite(e: &hbb_common::anyhow::Error) -> bool {
    if e.downcast_ref::<NotSent>().is_some() {
        return true;
    }
    match e.downcast_ref::<dbus::Error>() {
        Some(d) => !is_connection_error(d),
        None => false,
    }
}

/// The unit's ActiveState; None when systemd no longer knows it (stopped and collected). Two calls,
/// each bounded by `per_call`.
fn active_state(name: &str, per_call: Duration) -> ResultType<Option<String>> {
    let unit = with_manager(per_call, |m| {
        let (p,): (DbusPath<'static>,) = m.method_call(MANAGER, "GetUnit", (name,))?;
        Ok(p)
    });
    if dbus_err_is(&unit, &[NO_SUCH_UNIT]) {
        return Ok(None);
    }
    let state = with_proxy(unit?, per_call, |p| {
        use dbus::blocking::stdintf::org_freedesktop_dbus::Properties;
        p.get::<String>(UNIT_IFACE, "ActiveState")
    });
    if dbus_err_is(&state, &[UNKNOWN_OBJECT]) {
        return Ok(None); // collected between the two calls
    }
    Ok(Some(state?))
}

fn is_down(state: &ResultType<Option<String>>) -> bool {
    matches!(state, Ok(None)) || matches!(state, Ok(Some(s)) if s == "inactive" || s == "failed")
}

/// Stop one of our units within STOP_DEADLINE (all calls included). True = gone.
fn stop_unit_by_name(name: &str) -> bool {
    if !is_our_unit_name(name) {
        log::error!("refusing to stop '{name}': not a server unit");
        return false;
    }
    // Every call below is bounded by what is left of this budget.
    let deadline = Instant::now() + STOP_DEADLINE;
    let kill_phase = deadline - Duration::from_secs(1); // 1 s kept for the SIGKILL path
    let res = with_manager(STOP_CALL.min(remaining(kill_phase)), |m| {
        let (_job,): (DbusPath,) = m.method_call(MANAGER, "StopUnit", (name, "replace"))?;
        Ok(())
    });
    if dbus_err_is(&res, &[NO_SUCH_UNIT]) {
        return true;
    }
    if let Err(e) = res {
        log::error!("{name}: StopUnit failed: {e}");
    }
    if wait_down(name, kill_phase) {
        log::info!("stopped {name}");
        return true;
    }
    // systemd SIGKILLs the cgroup after TimeoutStopSec on its own; this covers a stop job that
    // never ran (e.g. StopUnit itself failed).
    log::error!("{name}: still up; sending SIGKILL to the whole unit");
    let left = remaining(deadline);
    if !left.is_zero() {
        let _ = with_manager(STOP_CALL.min(left), |m| {
            m.method_call::<(), _, _, _>(MANAGER, "KillUnit", (name, "all", 9i32))
        });
    }
    if wait_down(name, deadline) {
        return true;
    }
    log::error!("{name}: could not be confirmed stopped; will retry");
    false
}

/// Poll until the unit is down or `deadline`; each probe is bounded by the time left. Units are
/// CollectMode=inactive-or-failed, so a down unit disappears on its own (no ResetFailedUnit needed).
fn wait_down(name: &str, deadline: Instant) -> bool {
    loop {
        let left = remaining(deadline);
        if left.is_zero() {
            return false;
        }
        // Two calls per probe: give each half of what is left, at most PROBE_CALL.
        if is_down(&active_state(name, PROBE_CALL.min(left / 2))) {
            return true;
        }
        let left = remaining(deadline);
        if left.is_zero() {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50).min(left));
    }
}

/// Server units a previous service instance left behind. Nothing may start until they are
/// confirmed gone (the startup equivalent of a pending stop).
pub struct Leftovers {
    /// None = not listed yet.
    pending: Option<Vec<String>>,
    since: Instant,
    warned: bool,
}

impl Leftovers {
    pub fn new() -> Self {
        Leftovers {
            pending: None,
            since: Instant::now(),
            warned: false,
        }
    }

    /// Try to finish the startup cleanup; true once nothing is left (cheap after that).
    /// Listing that keeps failing (no bus) gives up after LIST_GIVE_UP so remote access still
    /// starts (the start then falls back to sudo, since no unit can be created without the bus).
    /// Units that are listed but will not stop block starting until they do.
    pub fn settle(&mut self) -> bool {
        if self.pending.is_none() {
            match list_server_units() {
                Ok(names) => {
                    for n in &names {
                        log::info!("startup: stopping leftover {n}");
                    }
                    self.pending = Some(names);
                }
                Err(e) if self.since.elapsed() >= LIST_GIVE_UP => {
                    log::error!(
                        "startup: cannot list leftover server units for {:?} ({e}); starting anyway",
                        self.since.elapsed()
                    );
                    self.pending = Some(Vec::new());
                }
                Err(e) => {
                    log::warn!("startup: cannot list leftover server units yet: {e}");
                    return false;
                }
            }
        }
        let pending = self.pending.as_mut().unwrap();
        pending.retain(|n| !stop_unit_by_name(n));
        if !pending.is_empty() && !self.warned {
            log::error!("startup: leftover units not stopped yet, not starting a server: {pending:?}");
            self.warned = true;
        }
        pending.is_empty()
    }
}

fn list_server_units() -> ResultType<Vec<String>> {
    with_manager(START_CALL, |m| {
        type Row = (
            String,
            String,
            String,
            String,
            String,
            String,
            DbusPath<'static>,
            u32,
            String,
            DbusPath<'static>,
        );
        let (rows,): (Vec<Row>,) = m.method_call(
            MANAGER,
            "ListUnitsByPatterns",
            (
                Vec::<String>::new(),
                vec![format!("{UNIT_PREFIX}*.service")],
            ),
        )?;
        Ok(rows
            .into_iter()
            .map(|r| r.0)
            .filter(|n| is_our_unit_name(n))
            .collect::<Vec<String>>())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_names_are_ours_and_bounded() {
        let n = unit_name(1000, 68800, 3);
        assert_eq!(n, "rustdesk-server-1000-68800-3.service");
        assert!(is_our_unit_name(&n));
        for bad in [
            "rustdesk.service",
            "rustdesk-server-.service",
            "rustdesk-server-1000.scope",
            "rustdesk-server-1000-x.service",
            "rustdesk-server-1000/../x.service",
            "sshd.service",
            "rustdesk-server-1000-1.service.d",
        ] {
            assert!(!is_our_unit_name(bad), "{bad}");
        }
    }

    #[test]
    fn uids_are_parsed_strictly() {
        assert_eq!(parse_uid("1000"), Some(1000));
        assert_eq!(parse_uid("115"), Some(115));
        for bad in [
            "",
            "0",
            "00",
            "-1",
            "4294967295",
            "4294967296",
            "1e3",
            " 1000",
            "1000\n",
            "root",
        ] {
            assert_eq!(parse_uid(bad), None, "{bad:?}");
        }
    }

    /// Needs root and a running systemd (rdmdev host, not the build container):
    ///   sudo <test-binary> --ignored --exact platform::linux_server_unit::tests::stop_kills_the_whole_tree
    /// A tree with a background child and a setsid escapee (what --server/--tray/--cm look like to
    /// the kernel) must be completely gone after stop(). Negative control: the same tree under
    /// `sudo -u nobody` with sudo SIGKILLed leaves all three sleeps running.
    #[test]
    #[ignore]
    fn stop_kills_the_whole_tree() {
        let unit = ServerUnit::start(
            "65534",
            Path::new("/bin/sh"),
            &["-c", "sleep 300 & setsid sleep 300 & exec sleep 300"],
            &[],
        )
        .unwrap_or_else(|_| panic!("start"));
        std::thread::sleep(Duration::from_millis(500));
        let pids = |name: &str| -> Vec<u32> {
            let cg = format!("/sys/fs/cgroup/system.slice/{name}/cgroup.procs");
            std::fs::read_to_string(cg)
                .unwrap_or_default()
                .lines()
                .filter_map(|l| l.trim().parse().ok())
                .collect()
        };
        let before = pids(unit.name());
        assert!(
            before.len() >= 3,
            "expected the tree in the unit's cgroup, got {before:?}"
        );
        assert!(!unit.has_exited());
        assert!(unit.stop());
        assert!(unit.has_exited());
        for pid in before {
            assert!(
                !Path::new(&format!("/proc/{pid}")).exists()
                    || std::fs::read_to_string(format!("/proc/{pid}/status"))
                        .map_or(true, |s| s.contains("State:\tZ")),
                "pid {pid} survived stop()"
            );
        }
        // An intentional stop is not a failed start.
        assert!(!ServerUnit::degraded());
    }

    /// Root + systemd, like above: the environment and limits the unit's process actually gets.
    #[test]
    #[ignore]
    fn unit_process_gets_the_built_environment_and_limits() {
        let unit = ServerUnit::start(
            "65534",
            Path::new("/bin/sh"),
            &["-c", "exec sleep 300"],
            &[
                ("DISPLAY".to_owned(), ":7".to_owned()),
                ("PATH".to_owned(), "/tmp/evil".to_owned()),
            ],
        )
        .unwrap_or_else(|_| panic!("start"));
        std::thread::sleep(Duration::from_millis(500));
        let cg = format!("/sys/fs/cgroup/system.slice/{}/cgroup.procs", unit.name());
        let pid: u32 = std::fs::read_to_string(cg)
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let env = std::fs::read(format!("/proc/{pid}/environ")).unwrap();
        let env: Vec<String> = env
            .split(|b| *b == 0)
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        let limits = std::fs::read_to_string(format!("/proc/{pid}/limits")).unwrap();
        assert!(unit.stop());
        assert!(env.contains(&"DISPLAY=:7".to_owned()), "{env:?}");
        assert!(env.contains(&format!("PATH={SERVER_PATH}")), "{env:?}");
        assert!(
            env.contains(&"XDG_RUNTIME_DIR=/run/user/65534".to_owned()),
            "{env:?}"
        );
        let nofile = limits
            .lines()
            .find(|l| l.starts_with("Max open files"))
            .unwrap();
        assert!(nofile.contains("1048576"), "{nofile}");
        // Login environment from the passwd entry (SetLoginEnvironment). systemd leaves SHELL unset for
        // a nologin shell, which nobody has; a real user gets it.
        assert!(env.contains(&"LOGNAME=nobody".to_owned()), "{env:?}");
        assert!(env.contains(&"HOME=/nonexistent".to_owned()), "{env:?}");
        assert!(
            !env.iter()
                .any(|e| e.starts_with("SHELL=") && !e.ends_with("nologin")),
            "{env:?}"
        );
    }

    #[test]
    fn own_unit_parses_cgroup_v2() {
        assert_eq!(
            own_unit_from_cgroup("0::/system.slice/rustdesk.service\n").as_deref(),
            Some("rustdesk.service")
        );
        assert_eq!(
            own_unit_from_cgroup("0::/user.slice/user-1000.slice/session-2.scope\n"),
            None
        );
        assert_eq!(own_unit_from_cgroup("0::/\n"), None);
        assert_eq!(own_unit_from_cgroup("1:name=systemd:/x.service\n"), None);
    }

    #[test]
    fn environment_is_explicit_and_validated() {
        let envs = vec![
            ("DISPLAY".to_owned(), ":0".to_owned()),
            ("HOME".to_owned(), "/home/u".to_owned()),
            ("BAD-KEY".to_owned(), "x".to_owned()),
            ("TERM".to_owned(), "a\nb".to_owned()),
            ("DISPLAY".to_owned(), ":9".to_owned()),
            ("PATH".to_owned(), "/tmp/evil".to_owned()),
            ("XDG_RUNTIME_DIR".to_owned(), "/tmp/evil".to_owned()),
        ];
        let inherited = vec![
            ("PIPEWIRE_LATENCY".to_owned(), "1024/48000".to_owned()),
            ("LANG".to_owned(), "en_US.UTF-8".to_owned()),
            ("LC_TIME".to_owned(), "en_GB.UTF-8".to_owned()),
            ("LD_PRELOAD".to_owned(), "/tmp/x.so".to_owned()),
            ("INVOCATION_ID".to_owned(), "abc".to_owned()),
        ];
        let env = build_environment(1000, &envs, &inherited);
        assert!(env.contains(&"DISPLAY=:0".to_owned()));
        assert!(!env.iter().any(|e| e == "DISPLAY=:9"));
        assert!(!env.iter().any(|e| e.starts_with("BAD-KEY")));
        assert!(!env.iter().any(|e| e.starts_with("TERM=")));
        assert!(env.contains(&"XDG_RUNTIME_DIR=/run/user/1000".to_owned()));
        assert!(!env.iter().any(|e| e.ends_with("/tmp/evil")));
        assert!(env.contains(&format!("PATH={SERVER_PATH}")));
        assert!(env.contains(&"PIPEWIRE_LATENCY=1024/48000".to_owned()));
        assert!(env.contains(&"LANG=en_US.UTF-8".to_owned()));
        assert!(env.contains(&"LC_TIME=en_GB.UTF-8".to_owned()));
        assert!(!env
            .iter()
            .any(|e| e.starts_with("LD_PRELOAD") || e.starts_with("INVOCATION_ID")));
    }

    #[test]
    fn pam_limits_apply_by_user_star_and_group_with_units() {
        let conf = "\
# comment
@pipewire   - rtprio  95
@pipewire   - nice    -19
@pipewire   - memlock 4194304
*           soft core 0
alice       hard nofile 4096
bob         -    nofile 9999
alice       soft nofile 2048   # trailing comment
*           -    bogus  1
*           -    nproc  notanumber
"
        .to_owned();
        let later = "alice soft nofile 3000\n".to_owned();
        let get = |user: &str, groups: &[&str]| {
            let g: Vec<String> = groups.iter().map(|s| s.to_string()).collect();
            pam_limits_for(&[conf.clone(), later.clone()], user, 1000, &g, 1 << 20)
        };
        let alice = get("alice", &[]);
        assert!(
            alice.contains(&("LimitNOFILE", Some(3000), Some(4096))),
            "{alice:?}"
        );
        assert!(alice.contains(&("LimitCORE", Some(0), None)), "{alice:?}");
        assert!(!alice
            .iter()
            .any(|(p, _, _)| *p == "LimitRTPRIO" || *p == "LimitNPROC"));
        let pw = get("carol", &["pipewire"]);
        assert!(pw.contains(&("LimitRTPRIO", Some(95), Some(95))), "{pw:?}");
        assert!(pw.contains(&("LimitNICE", Some(39), Some(39))), "{pw:?}");
        assert!(
            pw.contains(&("LimitMEMLOCK", Some(4194304 * 1024), Some(4194304 * 1024))),
            "{pw:?}"
        );
        assert!(!get("bob", &[]).iter().any(|(p, _, _)| *p == "LimitRTPRIO"));
    }

    #[test]
    fn pam_limits_specific_beats_later_wildcard_and_sides_stay_separate() {
        let conf = "\
alice   -    nofile 4096
*       -    nofile 1024
@staff  hard nofile 2048
*       soft core   0
bob     hard core   100
1000:1999 - nproc 500
*       -    nproc 50
"
        .to_owned();
        let later = "*    -  nofile 999\n".to_owned();
        let g = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let alice = pam_limits_for(&[conf.clone(), later.clone()], "alice", 1500, &g(&["staff"]), 1 << 20);
        // user line beats the group line and both later '*' lines, on both sides
        assert!(alice.contains(&("LimitNOFILE", Some(4096), Some(4096))), "{alice:?}");
        // soft-only line sets only the soft side
        assert!(alice.contains(&("LimitCORE", Some(0), None)), "{alice:?}");
        // uid range is user-class: beats the later '*'
        assert!(alice.contains(&("LimitNPROC", Some(500), Some(500))), "{alice:?}");
        let carol = pam_limits_for(&[conf.clone(), later.clone()], "carol", 3000, &g(&["staff"]), 1 << 20);
        // group (hard) beats '*' on the hard side; the later '*' still sets the soft side
        assert!(carol.contains(&("LimitNOFILE", Some(999), Some(2048))), "{carol:?}");
        assert!(carol.contains(&("LimitNPROC", Some(50), Some(50))), "{carol:?}");
        let bob = pam_limits_for(&[conf], "bob", 3000, &[], 1 << 20);
        assert!(bob.contains(&("LimitCORE", Some(0), Some(100 * 1024))), "{bob:?}");
    }

    #[test]
    fn pam_limits_special_values() {
        let conf = "\
a - nice -1
b - nice unlimited
c - nice -40
d - nofile unlimited
e - memlock unlimited
f - rtprio -1
"
        .to_owned();
        let one = |u: &str| pam_limits_for(&[conf.clone()], u, 1000, &[], 1_048_576);
        assert!(one("a").contains(&("LimitNICE", Some(21), Some(21))));
        assert!(one("b").contains(&("LimitNICE", Some(21), Some(21))));
        assert!(one("c").contains(&("LimitNICE", Some(40), Some(40)))); // clamped to -20
        assert!(one("d").contains(&("LimitNOFILE", Some(1_048_576), Some(1_048_576))));
        assert!(one("e").contains(&("LimitMEMLOCK", Some(u64::MAX), Some(u64::MAX))));
        assert!(one("f").contains(&("LimitRTPRIO", Some(u64::MAX), Some(u64::MAX))));
    }

    #[test]
    fn limit_props_map_sides_and_default_nofile() {
        let props = limit_props_from(vec![
            ("LimitCORE", Some(0), None),
            ("LimitSTACK", None, Some(8192)),
            ("LimitNPROC", Some(900), Some(500)),
        ]);
        let has = |k: &str, v: u64| props.contains(&(k.to_owned(), v));
        assert!(has("LimitCORESoft", 0) && !props.iter().any(|(k, _)| k == "LimitCORE"));
        assert!(has("LimitSTACK", 8192) && !props.iter().any(|(k, _)| k == "LimitSTACKSoft"));
        assert!(has("LimitNPROC", 500) && has("LimitNPROCSoft", 500)); // soft clamped to hard
        assert!(has("LimitNOFILE", SUDO_NOFILE) && has("LimitNOFILESoft", SUDO_NOFILE));
        // a pam soft-only nofile keeps its soft value; the hard side is the sudo value
        let p2 = limit_props_from(vec![("LimitNOFILE", Some(4096), None)]);
        assert!(p2.contains(&("LimitNOFILESoft".to_owned(), 4096)));
        assert!(p2.contains(&("LimitNOFILE".to_owned(), SUDO_NOFILE)));
    }

    #[test]
    fn systemd_versions_parse() {
        assert_eq!(parse_systemd_version("255.4-1ubuntu8.17"), Some(255));
        assert_eq!(parse_systemd_version("259"), Some(259));
        assert_eq!(parse_systemd_version("v249.11"), Some(249));
        assert_eq!(parse_systemd_version(""), None);
        assert_eq!(parse_systemd_version("abc"), None);
    }

    #[test]
    fn start_errors_are_definite_only_when_systemd_cannot_have_acted() {
        let de = |name: &str| {
            hbb_common::anyhow::Error::from(dbus::Error::new_custom(name, "x"))
        };
        assert!(start_error_is_definite(&de("org.freedesktop.DBus.Error.InvalidArgs")));
        assert!(start_error_is_definite(&de("org.freedesktop.DBus.Error.AccessDenied")));
        assert!(start_error_is_definite(&hbb_common::anyhow::Error::from(NotSent(
            dbus::Error::new_custom("org.freedesktop.DBus.Error.FileNotFound", "x")
        ))));
        assert!(!start_error_is_definite(&de("org.freedesktop.DBus.Error.NoReply")));
        assert!(!start_error_is_definite(&de("org.freedesktop.DBus.Error.Timeout")));
        assert!(!start_error_is_definite(&de("org.freedesktop.DBus.Error.Disconnected")));
        assert!(!start_error_is_definite(&hbb_common::anyhow::anyhow!("other")));
    }
}
