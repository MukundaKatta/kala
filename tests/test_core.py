"""Tests for Kala."""
from src.core import Kala
def test_init(): assert Kala().get_stats()["ops"] == 0
def test_op(): c = Kala(); c.search(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Kala(); [c.search() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Kala(); c.search(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Kala(); r = c.search(); assert r["service"] == "kala"
