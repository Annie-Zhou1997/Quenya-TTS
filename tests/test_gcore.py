"""Tests for quenya_tts.gcore, with a fake Gcore client: no account or machine needed."""

import base64
import datetime
import subprocess
import unittest
from types import SimpleNamespace as NS

from quenya_tts import gcore


def flavor(name, model, count, disabled=False):
    return NS(name=name, disabled=disabled, hardware_properties=NS(gpu_model=model, gpu_count=count))


def image(id, distro, version, driver="550", status="active", day=1):
    return NS(id=id, name=f"{distro}-{version}", os_distro=distro, os_version=version, gpu_driver=driver,
              status=status, created_at=datetime.datetime(2026, 1, day))


class FakeClusters:
    def __init__(self, fail_poll=False):
        self.created, self.deleted, self.fail_poll = [], [], fail_poll
        self.servers = NS(list=lambda cluster_id: [NS(ip_addresses=["2a03::1", "203.0.113.7"])])

    def create(self, **kwargs):
        self.created.append(kwargs)
        return NS(tasks=["task-1"])

    def list(self):
        return [NS(id="c-1", name=self.created[-1]["name"])] if self.created else []

    def delete_and_poll(self, cluster_id, **kwargs):
        self.deleted.append((cluster_id, kwargs))


class FakeClient:
    def __init__(self, fail_poll=False):
        self.clusters = FakeClusters()
        poll = (lambda task: (_ for _ in ()).throw(TimeoutError("still creating"))) if fail_poll else \
            (lambda task: NS(created_resources=NS(clusters=["c-1"])))
        self.cloud = NS(gpu_virtual=NS(clusters=self.clusters), tasks=NS(poll=poll))


class PickTest(unittest.TestCase):

    def test_flavor_with_one_enabled_gpu_of_the_model(self):
        flavors = [flavor("8xh100", "H100", 8), flavor("1xh100-off", "H100", 1, disabled=True),
                   flavor("1xl40s", "L40S", 1), flavor("1xh100", "NVIDIA H100", 1)]
        self.assertEqual(gcore.pick_flavor(flavors, "H100"), "1xh100")
        self.assertEqual(gcore.pick_flavor(flavors, "L40S"), "1xl40s")
        self.assertIsNone(gcore.pick_flavor(flavors, "A100"))

    def test_newest_ubuntu_image_with_a_gpu_driver(self):
        images = [image("old", "ubuntu", "22.04"), image("new", "ubuntu", "24.04"),
                  image("nodriver", "ubuntu", "26.04", driver=None), image("deb", "debian", "13")]
        self.assertEqual(gcore.pick_image(images), "new")
        self.assertIsNone(gcore.pick_image([image("deb", "debian", "13")]))


class RegionTest(unittest.TestCase):

    def test_only_eu_regions(self):
        for country in ("LU", "Luxembourg", "pt", "Finland"):
            self.assertTrue(gcore.in_eu(country), country)
        for country in ("GB", "United Kingdom", "KR", "South Korea", "", None):
            self.assertFalse(gcore.in_eu(country), country)


class ScriptsTest(unittest.TestCase):

    def test_machine_powers_off_at_the_limit_from_boot(self):
        self.assertIn("shutdown -P +390 ", gcore.boot_script(6.5))

    def test_settings_are_quoted(self):
        text = gcore.run_input("echo run\n", {"STORAGE_SECRET_ACCESS_KEY": "a'b; rm -rf /", "STEPS": 10})
        self.assertIn("export STEPS=10\n", text)
        self.assertIn("export STORAGE_SECRET_ACCESS_KEY='a'\"'\"'b; rm -rf /'\n", text)
        self.assertTrue(text.endswith("run() {\necho run\n\n}\nrun < /dev/null\n"))

    def test_watchdog_deletes_the_machine_with_its_disk_and_ip(self):
        text = gcore.watchdog_script("key'1", 11, 22, "c-1", 3600)
        self.assertIn("/cloud/v3/gpu/virtual/11/22/clusters/c-1?all_floating_ips=true", text)
        self.assertIn("all_volumes=true", text)
        self.assertIn("--on-active=3600", text)
        self.assertNotIn("key'1 ", text)  # quoted, not bare

    def test_cleanup_only_touches_our_machines(self):
        clusters = [NS(name="quenya-tts-20261018-2200"), NS(name="other")]
        self.assertEqual([c.name for c in gcore.left_over(clusters)], ["quenya-tts-20261018-2200"])


class MachineTest(unittest.TestCase):

    def test_creates_and_always_deletes(self):
        client = FakeClient()
        with self.assertRaises(RuntimeError):
            with gcore.Machine(client, "1xh100", "img", "my-mac", max_hours=6) as machine:
                self.assertEqual(machine.ip, "203.0.113.7")
                raise RuntimeError("training failed")
        created = client.clusters.created[0]
        self.assertTrue(created["name"].startswith(gcore.NAME_PREFIX))
        self.assertEqual(created["tags"], gcore.TAGS)
        settings = created["servers_settings"]
        self.assertTrue(settings["volumes"][0]["delete_on_termination"])
        self.assertIn("shutdown -P +360", base64.b64decode(settings["user_data"]).decode())
        self.assertEqual(client.clusters.deleted,
                         [("c-1", {"all_floating_ips": True, "all_reserved_fixed_ips": True, "all_volumes": True})])

    def test_deletes_a_machine_that_failed_to_come_up(self):
        client = FakeClient(fail_poll=True)
        with self.assertRaises(TimeoutError):
            with gcore.Machine(client, "1xh100", "img", "my-mac", max_hours=6):
                self.fail("should not get here")
        self.assertEqual([cluster_id for cluster_id, _ in client.clusters.deleted], ["c-1"])


class RunScriptTest(unittest.TestCase):

    def test_bash_syntax(self):
        with open(gcore.RUN_SCRIPT) as f:
            subprocess.run(["bash", "-n"], input=gcore.run_input(f.read(), {"STEPS": 1}), text=True, check=True)

    def test_a_command_reading_stdin_does_not_swallow_the_script(self):
        script = gcore.run_input("cat\necho after\n", {"X": 1})
        result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout, "after\n")


if __name__ == "__main__":
    unittest.main()
