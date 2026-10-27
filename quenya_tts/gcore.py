"""
Trains the public model on a Gcore GPU machine in the EU, from your own computer, and makes sure the
machine never outlives the run: it is created for one training and deleted afterwards, with its disk and IP,
whether training worked, failed or was interrupted.

Every run has a time limit (--max-hours, 6 by default), enforced three times:
- here: when the limit is reached the run is stopped and the machine deleted;
- on the machine: it powers itself off at the limit, even if this computer sleeps or loses its connection;
- with --watchdog (a second Gcore API key kept only on the machine): the machine also deletes itself at the limit.
`cleanup` lists and deletes any Quenya-TTS machine that is still there, in case all of that failed.

The volunteer recordings go straight from the site's storage to the machine and are deleted with it; only the release (artificial
voice), logs and watermarked samples come back here, plus the checkpoint with --keep-checkpoint (it holds a
volunteer's voice: keep it on an encrypted disk, never publish it).

    pip install gcore
    export GCORE_API_KEY=...  GCORE_CLOUD_PROJECT_ID=...  GCORE_CLOUD_REGION_ID=...   # an EU region
    export STORAGE_ENDPOINT=...  STORAGE_REGION=...  STORAGE_ACCESS_KEY_ID=...  STORAGE_SECRET_ACCESS_KEY=...
    export STORAGE_KEY=...                                    # the site's settings (quenya-voices README)
    python -m quenya_tts.gcore flavors                                  # GPU machines and prices in the region
    python -m quenya_tts.gcore train --ssh-key-name my-mac -o ~/quenya-runs --steps 20000 --max-hours 6
    python -m quenya_tts.gcore cleanup                                  # anything left? (--yes deletes it)
"""

import argparse
import base64
import datetime
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time

NAME_PREFIX = "quenya-tts-"
TAGS = {"project": "quenya-tts"}
DEFAULT_MAX_HOURS = 6.0
# Kept back from the time limit for starting the machine before training, and for the release and copying the
# results back after it.
SETUP_MINUTES = 30
WRAP_UP_MINUTES = 20
API = "https://api.gcore.com"
# What the machine needs to read and decrypt the recordings: the same settings as the site's server.
STORAGE_SETTINGS = ("STORAGE_ENDPOINT", "STORAGE_REGION", "STORAGE_ACCESS_KEY_ID", "STORAGE_SECRET_ACCESS_KEY",
                    "STORAGE_KEY")
RUN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_run.sh")


# The Privacy Notice promises volunteers that their recordings stay in the European Union: the script refuses
# any other region. ISO codes and English names, as Gcore may report either.
EU_COUNTRIES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus", "CZ": "Czechia",
    "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France", "DE": "Germany", "GR": "Greece",
    "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg",
    "MT": "Malta", "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
    "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
}


def in_eu(country):
    """Whether a Gcore region's country (ISO code or English name) is in the European Union."""
    country = (country or "").strip().lower()
    return any(country in (code.lower(), name.lower()) for code, name in EU_COUNTRIES.items())


def pick_flavor(flavors, gpu="H100", count=1):
    """The name of the first enabled flavor with count GPUs of the given model, or None."""
    for flavor in flavors:
        hardware = flavor.hardware_properties
        if (not flavor.disabled and hardware.gpu_count == count
                and gpu.lower() in (hardware.gpu_model or "").lower()):
            return flavor.name
    return None


def pick_image(images):
    """The id of the newest Ubuntu image with a GPU driver, or None."""
    ubuntu = [image for image in images if "ubuntu" in (image.os_distro or image.name).lower()
              and image.status == "active" and image.gpu_driver]
    if not ubuntu:
        return None
    return max(ubuntu, key=lambda image: (image.os_version or "", image.created_at)).id


def boot_script(max_hours):
    """
    cloud-init user data that powers the machine off at the time limit, from the moment it boots: the last line
    of defence if this computer goes away before it can delete the machine. Needs no key.
    """
    minutes = int(max_hours * 60)
    return f"#!/bin/sh\nshutdown -P +{minutes} 'Quenya-TTS: time limit reached'\n"


def watchdog_script(api_key, project_id, region_id, cluster_id, seconds):
    """Commands (run on the machine over SSH) that make it delete itself after seconds, with its disk and IP."""
    url = (f"{API}/cloud/v3/gpu/virtual/{project_id}/{region_id}/clusters/{cluster_id}"
           "?all_floating_ips=true&all_reserved_fixed_ips=true&all_volumes=true")
    delete = f"curl -sS -X DELETE -H @/root/.gcore-watchdog {shlex.quote(url)}"
    return (f"printf 'Authorization: APIKey %s\\n' {shlex.quote(api_key)} "
            "| sudo install -m 600 /dev/stdin /root/.gcore-watchdog\n"
            f"sudo systemd-run --unit quenya-watchdog --on-active={int(seconds)} /bin/sh -c {shlex.quote(delete)}\n")


def run_input(script, settings):
    """
    What is sent to the machine's shell: the settings as exports, then the run script. The script is wrapped in a
    function, so that the shell reads all of it before running anything, and runs it with stdin closed: a command
    reading stdin would otherwise swallow the rest of the script.
    """
    exports = "".join(f"export {name}={shlex.quote(str(value))}\n" for name, value in settings.items())
    return f"{exports}run() {{\n{script}\n}}\nrun < /dev/null\n"


def left_over(clusters):
    """The Quenya-TTS machines among the clusters."""
    return [cluster for cluster in clusters if cluster.name.startswith(NAME_PREFIX)]


class Machine:
    """One GPU machine for one run. Use as a context manager: leaving it deletes the machine, however it is left."""

    def __init__(self, client, flavor, image, ssh_key_name, max_hours, user="ubuntu", disk_gb=200):
        self.client, self.flavor, self.image = client, flavor, image
        self.ssh_key_name, self.max_hours, self.user, self.disk_gb = ssh_key_name, max_hours, user, disk_gb
        self.cluster_id = None
        self.ip = None

    def __enter__(self):
        name = NAME_PREFIX + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Creating {name} ({self.flavor})...", flush=True)
        response = self.client.cloud.gpu_virtual.clusters.create(
            flavor=self.flavor, name=name, servers_count=1, tags=TAGS,
            servers_settings={
                "interfaces": [{"type": "external", "ip_family": "ipv4"}],
                "volumes": [{"source": "image", "image_id": self.image, "boot_index": 0, "name": f"{name}-boot",
                             "size": self.disk_gb, "type": "ssd_hiiops", "delete_on_termination": True}],
                "credentials": {"ssh_key_name": self.ssh_key_name},
                "user_data": base64.b64encode(boot_script(self.max_hours).encode()).decode(),
            })
        task = None
        try:
            task = self.client.cloud.tasks.poll(response.tasks[0])
            self.cluster_id = self._created_id(name, task)
            servers = list(self.client.cloud.gpu_virtual.clusters.servers.list(self.cluster_id))
            self.ip = next(ip for ip in servers[0].ip_addresses if ":" not in ip)
        except BaseException:
            # __exit__ does not run when __enter__ fails: find the machine and delete it here
            self.cluster_id = self.cluster_id or self._created_id(name, task)
            self.delete()
            raise
        print(f"Machine ready at {self.ip}", flush=True)
        return self

    def _created_id(self, name, task):
        if task is not None and task.created_resources and task.created_resources.clusters:
            return task.created_resources.clusters[0]
        for cluster in self.client.cloud.gpu_virtual.clusters.list():
            if cluster.name == name:
                return cluster.id
        return None

    def __exit__(self, *exc):
        self.delete()
        return False

    def delete(self):
        if self.cluster_id is None:
            return
        print(f"Deleting the machine {self.cluster_id}...", flush=True)
        try:
            self.client.cloud.gpu_virtual.clusters.delete_and_poll(
                self.cluster_id, all_floating_ips=True, all_reserved_fixed_ips=True, all_volumes=True)
            print("Machine deleted.", flush=True)
            self.cluster_id = None
        except Exception as err:
            print(f"\n!!! Could not delete the machine {self.cluster_id}: {err}\n"
                  "!!! It is still being billed. Run: python -m quenya_tts.gcore cleanup --yes\n", file=sys.stderr)
            raise


def ssh_command(user, ip, known_hosts, *remote):
    return ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", f"{user}@{ip}", *remote]


def wait_for_ssh(user, ip, known_hosts, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        if subprocess.run(ssh_command(user, ip, known_hosts, "true"), capture_output=True).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError(f"no SSH connection to {ip} after {timeout // 60} min (does the security group allow port 22?)")


def stay_awake():
    """On a Mac, keeps it from sleeping while this process runs."""
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def train(args, client):
    for name in STORAGE_SETTINGS:
        if not os.environ.get(name):
            sys.exit(f"Set {name} (the recording site's storage settings, see the quenya-voices README).")
    region = client.cloud.regions.get(client.cloud_region_id)
    if not in_eu(region.country):
        sys.exit(f"{region.display_name} ({region.country}) is outside the European Union: volunteer recordings "
                 "may only be trained on in the EU. Set GCORE_CLOUD_REGION_ID to an EU region.")
    flavor = args.flavor or pick_flavor(list(client.cloud.gpu_virtual.clusters.flavors.list(hide_disabled=True)),
                                        args.gpu)
    image = args.image or pick_image(list(client.cloud.gpu_virtual.clusters.images.list()))
    if not flavor or not image:
        sys.exit(f"No {'flavor with one ' + args.gpu if not flavor else 'Ubuntu GPU image'} in this region; "
                 "see `python -m quenya_tts.gcore flavors`, or pass --flavor / --image.")
    started = time.time()
    deadline = started + args.max_hours * 3600
    output = os.path.join(os.path.expanduser(args.output), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    stay_awake()
    # Stopping this script (Ctrl-C, closing the terminal, kill) must still delete the machine.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit("Stopped."))

    with tempfile.TemporaryDirectory() as tmp, Machine(client, flavor, image, args.ssh_key_name, args.max_hours,
                                                      args.user) as machine:
        known_hosts = os.path.join(tmp, "known_hosts")
        wait_for_ssh(args.user, machine.ip, known_hosts)
        if args.watchdog:
            key = os.environ.get("GCORE_WATCHDOG_API_KEY")
            if not key:
                sys.exit("--watchdog needs GCORE_WATCHDOG_API_KEY (a second API key with an expiry date).")
            subprocess.run(ssh_command(args.user, machine.ip, known_hosts, "sh"), check=True, text=True,
                           input=watchdog_script(key, client.cloud_project_id, client.cloud_region_id,
                                                 machine.cluster_id, deadline - time.time()))
        remaining = deadline - time.time()
        settings = {
            "BRANCH": args.branch, "STEPS": args.steps, "BATCH_SIZE": args.batch_size,
            "TRAIN_SECONDS": max(60, int(remaining - (SETUP_MINUTES + WRAP_UP_MINUTES) * 60)),
            "KEEP_CHECKPOINT": int(args.keep_checkpoint),
            **{name: os.environ[name] for name in STORAGE_SETTINGS},
        }
        with open(RUN_SCRIPT, encoding="utf8") as f:
            script = run_input(f.read(), settings)
        try:
            # the script arrives on stdin, so the storage keys are never in a file or on a command line on the machine
            result = subprocess.run(ssh_command(args.user, machine.ip, known_hosts, "bash", "-s"), text=True,
                                    input=script, timeout=max(60, deadline - time.time() - WRAP_UP_MINUTES * 60 / 2))
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            print("Time limit reached: stopping the run.", file=sys.stderr)
            ok = False
        os.makedirs(output, exist_ok=True)
        subprocess.run(["scp", "-r", "-q", "-o", "StrictHostKeyChecking=accept-new",
                        "-o", f"UserKnownHostsFile={known_hosts}", f"{args.user}@{machine.ip}:run/out/.", output],
                       timeout=WRAP_UP_MINUTES * 60 / 2)
    hours = (time.time() - started) / 3600
    print(f"{'Done' if ok else 'The run failed'} after {hours:.1f} h; results in {output}")
    if not ok:
        sys.exit(1)


def flavors(args, client):
    for flavor in client.cloud.gpu_virtual.clusters.flavors.list(include_prices=True):
        hardware = flavor.hardware_properties
        price = getattr(flavor, "price", None)
        cost = f"{price.price_per_hour} {price.currency_code}/h" if price and price.price_per_hour else ""
        state = "disabled" if flavor.disabled else f"{flavor.capacity} free"
        print(f"{flavor.name:40} {hardware.gpu_count}x {hardware.gpu_model or '?':12} {cost:16} {state}")


def cleanup(args, client):
    machines = left_over(client.cloud.gpu_virtual.clusters.list())
    if not machines:
        print("No Quenya-TTS machine is running.")
        return
    for cluster in machines:
        hours = (datetime.datetime.now(datetime.timezone.utc) - cluster.created_at).total_seconds() / 3600
        print(f"{cluster.name} ({cluster.id}): {cluster.status}, created {hours:.1f} h ago")
    if not args.yes:
        print("Run again with --yes to delete them.")
        return
    for cluster in machines:
        client.cloud.gpu_virtual.clusters.delete_and_poll(
            cluster.id, all_floating_ips=True, all_reserved_fixed_ips=True, all_volumes=True)
        print(f"Deleted {cluster.name}.")


def main():
    parser = argparse.ArgumentParser(description="Train Quenya-TTS on a Gcore GPU machine that is deleted afterwards.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train", help="create a machine, train, copy the results back, delete the machine")
    run.add_argument("--ssh-key-name", required=True, help="the name of your SSH key in Gcore (Cloud > SSH keys)")
    run.add_argument("-o", "--output", required=True, help="where the results go (a dated folder inside it)")
    run.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                     help="hard time limit for the whole run, machine included")
    run.add_argument("--steps", type=int, default=20000)
    run.add_argument("--batch_size", type=int, default=16)
    run.add_argument("--gpu", default="H100", help="GPU model to look for when --flavor is not given")
    run.add_argument("--flavor", help="Gcore flavor name (see the flavors command)")
    run.add_argument("--image", help="Gcore image id (default: the newest Ubuntu image with a GPU driver)")
    run.add_argument("--user", default="ubuntu", help="SSH user of the image")
    run.add_argument("--branch", default="ToucanTTS", help="Quenya-TTS branch to train with")
    run.add_argument("--keep-checkpoint", action="store_true",
                     help="also copy best.pt back (it holds a volunteer's voice: keep it private)")
    run.add_argument("--watchdog", action="store_true",
                     help="let the machine delete itself at the time limit (needs GCORE_WATCHDOG_API_KEY)")
    commands.add_parser("flavors", help="list the GPU machines of the region, with prices")
    clean = commands.add_parser("cleanup", help="list Quenya-TTS machines that are still there")
    clean.add_argument("--yes", action="store_true", help="delete them")
    args = parser.parse_args()
    if getattr(args, "max_hours", 1) <= (SETUP_MINUTES + WRAP_UP_MINUTES) / 60:
        parser.error(f"--max-hours must leave time for training after {SETUP_MINUTES + WRAP_UP_MINUTES} min of setup")

    from gcore import Gcore
    client = Gcore()
    {"train": train, "flavors": flavors, "cleanup": cleanup}[args.command](args, client)


if __name__ == "__main__":
    main()
