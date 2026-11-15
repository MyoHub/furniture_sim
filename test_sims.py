"""Smoke tests for loading and stepping all standalone and kitchen MuJoCo models."""

from __future__ import annotations

import os
import unittest

import mujoco
import numpy as np

model_paths = [
    "bin.xml",
    "boxcabinet.xml",
    "counters.xml",
    "dishwasher.xml",
    "hingecabinet.xml",
    "kettle.xml",
    "lightButtons.xml",
    "microwave.xml",
    "oven.xml",
    "simpleTable.xml",
    "slidecabinet.xml",
    "studyTable.xml",
    "ventionTable.xml",
    "kitchen/kitchen1.xml",
    "kitchen/kitchen2.xml",
    "kitchen/kitchen3.xml",
]


class Sim:
    """Lightweight simulation container pairing ``MjModel`` and ``MjData``."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)


class TestSims(unittest.TestCase):
    def get_sim(
        self,
        model_path: str | os.PathLike[str] | None = None,
        model_xmlstr: str | None = None,
    ) -> Sim:
        """Build a :class:`Sim` from ``model_path`` or ``model_xmlstr``."""
        if model_path is not None:
            path_str = os.fspath(model_path)
            if os.path.isabs(path_str):
                fullpath = path_str
            else:
                fullpath = os.path.join(os.path.dirname(__file__), path_str)
            if not os.path.exists(fullpath):
                raise FileNotFoundError(f"File {fullpath} does not exist")
            model = mujoco.MjModel.from_xml_path(fullpath)
        elif model_xmlstr is not None:
            model = mujoco.MjModel.from_xml_string(model_xmlstr)
        else:
            raise TypeError("Both model_path and model_xmlstr can't be None")

        return Sim(model)

    def test_sims(self) -> None:
        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                sim = self.get_sim(model_path)
                sim.forward()
                sim.step()
                self.assertFalse(np.isnan(sim.data.qpos).any())

    def test_get_sim_xmlstr_and_errors(self) -> None:
        xml = "<mujoco><worldbody><body><geom size='0.1'/></body></worldbody></mujoco>"
        sim = self.get_sim(model_xmlstr=xml)
        sim.step()
        self.assertEqual(sim.model.ngeom, 1)

        with self.assertRaises(FileNotFoundError):
            self.get_sim(model_path="does_not_exist.xml")

        with self.assertRaises(TypeError):
            self.get_sim()


if __name__ == "__main__":
    unittest.main()
