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
//! What `sudo` provided and is reproduced here (boostlap, 2026-09-29): the server's environment is
//! built from an explicit list (the caller's desktop variables + XDG_RUNTIME_DIR + PATH + a small
//! allow-list copied from the service's own environment); pam_env's files come in as EnvironmentFiles;
//! pam_umask's 022 as UMask. `PAMName=` is deliberately NOT set: it would open a logind session and
//! move the processes into a session scope, outside the unit's cgroup. sudo's PAM stack on Ubuntu 24.04
//! (common-session-noninteractive) has no pam_systemd, so today's server has no session either.
//! pam_himmelblau / pam_exec home-private (Entra) prepare a user's home at sign-in; the server only ever
//! starts for a session LightDM has already opened (or for the greeter user), so they already ran.

use dbus::{
    arg::{RefArg, Variant},
    blocking::{Connection, Proxy},
    Path as DbusPath,
};
use hbb_common::{bail, log, ResultType};
use std::{
    cell::RefCell,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const DEST: &str = "org.freedesktop.systemd1";
const OBJ: &str = "/org/freedesktop/systemd1";
const MANAGER: &str = "org.freedesktop.systemd1.Manager";
const UNIT_IFACE: &str = "org.freedesktop.systemd1.Unit";
const NO_SUCH_UNIT: &str = "org.freedesktop.systemd1.NoSuchUnit";
pub const UNIT_PREFIX: &str = "rustdesk-server-";
const CALL_TIMEOUT: Duration = Duration::from_secs(5);
/// TimeoutStopSec of the unit; systemd SIGKILLs the cgroup after it.
const STOP_TIMEOUT_USEC: u64 = 2_000_000;
/// How long `stop` waits for the unit to go away (stop timeout + margin).
const STOP_WAIT: Duration = Duration::from_millis(4_000);
/// sudo's secure_path on Ubuntu, minus /snap/bin (VNET OS removes snapd). The xrandr shim
/// (/usr/local/sbin/xrandr, Boost desktop/display/xrandr-rustdesk) relies on /usr/local/sbin first.
const SERVER_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
/// sudo -E passed the service's whole environment; these are the parts the server uses.
const INHERITED_ENV: &[&str] = &["PULSE_LATENCY_MSEC", "PIPEWIRE_LATENCY", "RUST_LOG"];
/// Open-files limit the sudo-started server got (pam_limits/sudo raise it to the hard limit).
const LIMIT_NOFILE: u64 = 1_048_576;

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A D-Bus variant of any argument type (for the a(sv) property list).
fn v<T: RefArg + 'static>(x: T) -> Variant<Box<dyn RefArg>> {
    Variant(Box::new(x))
}

thread_local! {
    // The service loop is single-threaded; one system-bus connection is reused across ticks.
    static BUS: RefCell<Option<Connection>> = RefCell::new(None);
}

fn with_manager<T>(f: impl FnOnce(&Proxy<&Connection>) -> Result<T, dbus::Error>) -> ResultType<T> {
    BUS.with(|bus| {
        let mut bus = bus.borrow_mut();
        if bus.is_none() {
            *bus = Some(Connection::new_system()?);
        }
        let conn = bus.as_ref().unwrap();
        let res = f(&conn.with_proxy(DEST, OBJ, CALL_TIMEOUT));
        if let Err(e) = &res {
            // A dropped bus connection surfaces as Disconnected/NoReply: reconnect next time.
            if e.name() != Some(NO_SUCH_UNIT) {
                *bus = None;
            }
        }
        Ok(res?)
    })
}

fn is_no_such_unit<T>(res: &ResultType<T>) -> bool {
    match res {
        Err(e) => e
            .downcast_ref::<dbus::Error>()
            .map_or(false, |e| e.name() == Some(NO_SUCH_UNIT)),
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

pub fn unit_name(uid: &str, service_pid: u32, seq: u64) -> String {
    format!("{UNIT_PREFIX}{uid}-{service_pid}-{seq}.service")
}

/// The `Environment=` list: caller-supplied variables (validated), XDG_RUNTIME_DIR, PATH, and the
/// allow-listed variables copied from `inherited`. Later duplicates of a key are dropped.
pub fn build_environment(
    uid: &str,
    envs: &[(String, String)],
    inherited: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    fn valid_key(k: &str) -> bool {
        let mut it = k.bytes();
        matches!(it.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_')
            && it.all(|c| c.is_ascii_alphanumeric() || c == b'_')
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut push = |k: &str, v: &str| {
        // systemd's Environment= takes raw KEY=VALUE strings over D-Bus (no quoting); a newline or NUL
        // in a value is rejected, not escaped.
        if !valid_key(k) || v.contains('\n') || v.contains('\0') || seen.iter().any(|s| s == k) {
            return;
        }
        seen.push(k.to_owned());
        out.push(format!("{k}={v}"));
    };
    // Fixed values first: a caller-supplied PATH or XDG_RUNTIME_DIR never overrides them.
    push("XDG_RUNTIME_DIR", &format!("/run/user/{uid}"));
    push("PATH", SERVER_PATH);
    for (k, v) in envs {
        push(k, v);
    }
    for k in INHERITED_ENV {
        if let Some(v) = inherited(k) {
            push(k, &v);
        }
    }
    out
}

pub struct ServerUnit {
    name: String,
}

impl ServerUnit {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Start `exe args...` as `uid` in a new transient unit.
    pub fn start(
        uid: &str,
        exe: &Path,
        args: &[&str],
        envs: &[(String, String)],
    ) -> ResultType<ServerUnit> {
        if uid.is_empty() || !uid.bytes().all(|b| b.is_ascii_digit()) || uid == "0" {
            bail!("refusing to start a server unit for uid '{uid}'");
        }
        let Some(exe_s) = exe.to_str() else {
            bail!("non-UTF-8 executable path");
        };
        let name = unit_name(uid, std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
        let mut argv = vec![exe_s.to_owned()];
        argv.extend(args.iter().map(|a| a.to_string()));
        let environment = build_environment(uid, envs, |k| std::env::var(k).ok());

        let mut props: Vec<(&str, Variant<Box<dyn RefArg>>)> = vec![
            ("Description", v(format!("RustDesk server for uid {uid}"))),
            ("ExecStart", v(vec![(exe_s.to_owned(), argv, false)])),
            ("User", v(uid.to_owned())),
            ("Environment", v(environment)),
            (
                "EnvironmentFiles",
                v(vec![
                    ("/etc/environment".to_owned(), true),
                    ("/etc/default/locale".to_owned(), true),
                ]),
            ),
            ("UMask", v(0o022u32)),
            ("LimitNOFILE", v(LIMIT_NOFILE)),
            ("KillMode", v("control-group".to_owned())),
            ("TimeoutStopUSec", v(STOP_TIMEOUT_USEC)),
            // Gone as soon as it stops or fails: no "failed" leftovers, names are never reused anyway.
            ("CollectMode", v("inactive-or-failed".to_owned())),
        ];
        if let Some(parent) = own_unit() {
            // Stopping rustdesk.service stops every server it started.
            props.push(("BindsTo", v(vec![parent.clone()])));
            props.push(("After", v(vec![parent])));
        }
        let aux: Vec<(&str, Vec<(&str, Variant<Box<dyn RefArg>>)>)> = vec![];
        with_manager(|m| {
            let (_job,): (DbusPath,) =
                m.method_call(MANAGER, "StartTransientUnit", (name.as_str(), "fail", props, aux))?;
            Ok(())
        })?;
        log::info!("started {name} (uid {uid})");
        Ok(ServerUnit { name })
    }

    /// The unit's ActiveState; None when systemd no longer knows it (stopped and collected).
    fn active_state(name: &str) -> ResultType<Option<String>> {
        let unit = with_manager(|m| {
            let (p,): (DbusPath,) = m.method_call(MANAGER, "GetUnit", (name,))?;
            Ok(p)
        });
        if is_no_such_unit(&unit) {
            return Ok(None);
        }
        let unit = unit?;
        let state: ResultType<String> = BUS.with(|bus| {
            let bus = bus.borrow();
            let Some(conn) = bus.as_ref() else {
                bail!("system bus not connected");
            };
            use dbus::blocking::stdintf::org_freedesktop_dbus::Properties;
            Ok(conn
                .with_proxy(DEST, unit, CALL_TIMEOUT)
                .get::<String>(UNIT_IFACE, "ActiveState")?)
        });
        match state {
            Ok(s) => Ok(Some(s)),
            // Collected between the two calls.
            Err(e) if e.downcast_ref::<dbus::Error>().map_or(false, |d| {
                d.name() == Some("org.freedesktop.DBus.Error.UnknownObject")
            }) =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// True once the server has exited (unit inactive, failed or gone). A bus error counts as
    /// "still running": restarting on a transient bus hiccup would start a second server.
    pub fn has_exited(&self) -> bool {
        match Self::active_state(&self.name) {
            Ok(None) => true,
            Ok(Some(s)) => s == "inactive" || s == "failed",
            Err(e) => {
                log::warn!("{}: cannot read state: {e}", self.name);
                false
            }
        }
    }

    /// Stop the unit and wait until every process in it is gone.
    pub fn stop(&self) {
        stop_unit_by_name(&self.name);
    }
}

fn stop_unit_by_name(name: &str) {
    if !is_our_unit_name(name) {
        log::error!("refusing to stop '{name}': not a server unit");
        return;
    }
    let res = with_manager(|m| {
        let (_job,): (DbusPath,) = m.method_call(MANAGER, "StopUnit", (name, "replace"))?;
        Ok(())
    });
    if is_no_such_unit(&res) {
        return;
    }
    if let Err(e) = res {
        log::error!("{name}: StopUnit failed: {e}");
    }
    if wait_gone(name, STOP_WAIT) {
        log::info!("stopped {name}");
        return;
    }
    // systemd should already have SIGKILLed the cgroup after TimeoutStopSec; make sure.
    log::error!("{name}: still active after {:?}; sending SIGKILL to the whole unit", STOP_WAIT);
    let _ = with_manager(|m| m.method_call::<(), _, _, _>(MANAGER, "KillUnit", (name, "all", 9i32)));
    if !wait_gone(name, Duration::from_secs(1)) {
        log::error!("{name}: could not be stopped");
    }
}

fn wait_gone(name: &str, limit: Duration) -> bool {
    let start = Instant::now();
    loop {
        match ServerUnit::active_state(name) {
            Ok(None) => return true,
            Ok(Some(s)) if s == "inactive" || s == "failed" => {
                let _ = with_manager(|m| m.method_call::<(), _, _, _>(MANAGER, "ResetFailedUnit", (name,)));
                return true;
            }
            _ => {}
        }
        if start.elapsed() >= limit {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Startup: stop every server unit a previous service instance left behind.
pub fn sweep() {
    let listed = with_manager(|m| {
        type Row = (
            String, String, String, String, String, String, DbusPath<'static>, u32, String,
            DbusPath<'static>,
        );
        let (rows,): (Vec<Row>,) = m.method_call(
            MANAGER,
            "ListUnitsByPatterns",
            (Vec::<String>::new(), vec![format!("{UNIT_PREFIX}*.service")]),
        )?;
        Ok(rows.into_iter().map(|r| r.0).collect::<Vec<String>>())
    });
    match listed {
        Ok(names) => {
            for n in names.iter().filter(|n| is_our_unit_name(n)) {
                log::info!("sweep: stopping leftover {n}");
                stop_unit_by_name(n);
            }
        }
        Err(e) => log::warn!("sweep: cannot list server units: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_names_are_ours_and_bounded() {
        let n = unit_name("1000", 68800, 3);
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

    /// Needs root and a running systemd (rdmdev host, not the build container):
    ///   sudo <test-binary> --ignored --exact platform::linux_server_unit::tests::stop_kills_the_whole_tree
    /// A tree with a background child and a setsid escapee (what --server/--tray/--cm look like to
    /// the kernel) must be completely gone after stop(). Negative control: the same tree under
    /// `sudo -u nobody` with sudo SIGKILLed leaves both sleeps running (see the PR's test log).
    #[test]
    #[ignore]
    fn stop_kills_the_whole_tree() {
        let unit = ServerUnit::start(
            "65534",
            Path::new("/bin/sh"),
            &["-c", "sleep 300 & setsid sleep 300 & exec sleep 300"],
            &[],
        )
        .expect("start");
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
        assert!(before.len() >= 3, "expected the tree in the unit's cgroup, got {before:?}");
        assert!(!unit.has_exited());
        unit.stop();
        assert!(unit.has_exited());
        for pid in before {
            assert!(
                !Path::new(&format!("/proc/{pid}")).exists()
                    || std::fs::read_to_string(format!("/proc/{pid}/status"))
                        .map_or(true, |s| s.contains("State:\tZ")),
                "pid {pid} survived stop()"
            );
        }
    }

    #[test]
    fn own_unit_parses_cgroup_v2() {
        assert_eq!(
            own_unit_from_cgroup("0::/system.slice/rustdesk.service\n").as_deref(),
            Some("rustdesk.service")
        );
        assert_eq!(own_unit_from_cgroup("0::/user.slice/user-1000.slice/session-2.scope\n"), None);
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
        ];
        let inherited = |k: &str| match k {
            "PIPEWIRE_LATENCY" => Some("1024/48000".to_owned()),
            "LD_PRELOAD" => Some("/tmp/x.so".to_owned()),
            _ => None,
        };
        let env = build_environment("1000", &envs, inherited);
        assert!(env.contains(&"DISPLAY=:0".to_owned()));
        assert!(!env.iter().any(|e| e == "DISPLAY=:9"));
        assert!(!env.iter().any(|e| e.starts_with("BAD-KEY")));
        assert!(!env.iter().any(|e| e.starts_with("TERM=")));
        assert!(env.contains(&"XDG_RUNTIME_DIR=/run/user/1000".to_owned()));
        assert!(env.contains(&"PIPEWIRE_LATENCY=1024/48000".to_owned()));
        assert!(!env.iter().any(|e| e.starts_with("LD_PRELOAD")));
        assert!(env.contains(&format!("PATH={SERVER_PATH}")));
        assert!(!env.iter().any(|e| e == "PATH=/tmp/evil"));
    }
}
