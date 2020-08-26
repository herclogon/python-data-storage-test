# Copyright (C) DATADVANCE, 2010-2020

"""pSeven typesystem usecases."""

import os
import shutil
import tempfile
import uuid

import numpy

from .. import convert, file
from .base_testcase import BaseTestCase


class Problem:
    """Optimization problem definition."""

    def __init__(
        self,
        variables,
        objectives,
        constraints,
        evaluate_objectives,
        evaluate_constraints,
    ):
        self.variables = variables
        self.objectives = objectives
        self.constraints = constraints
        self.evaluate_objectives = evaluate_objectives
        self.evaluate_constraints = evaluate_constraints

    @property
    def names(self):
        """Variables, objectives and constraints."""
        return self.variables_names + self.objectives_names + self.constraints_names

    @property
    def variables_names(self):
        """Problem variables."""
        return [x["name"] for x in self.variables]

    @property
    def objectives_names(self):
        """Problem objectives."""
        return [f["name"] for f in self.objectives]

    @property
    def objectives_slice(self):
        """Problem objectives slice."""
        return slice(len(self.variables), len(self.variables) + len(self.objectives))

    @property
    def constraints_names(self):
        """Problem constraints."""
        return [c["name"] for c in self.constraints]

    @property
    def constraints_slice(self):
        """Problem constraints slice."""
        return slice(
            len(self.variables) + len(self.objectives),
            len(self.variables) + len(self.objectives) + len(self.constraints),
        )

    @property
    def constraints_ub(self):
        """Constraints upper bounds."""
        return numpy.array([c["ub"] for c in self.constraints])

    @property
    def constraints_lb(self):
        """Constraints lower bounds."""
        return numpy.array([c["lb"] for c in self.constraints])


class DSX:
    """Design space exploration block."""

    def __init__(self, options):
        self._options = options

    def _before_solution(self, problem):
        raise NotImplementedError

    def _after_iteration(self, problem, x, f, c):
        raise NotImplementedError

    def _after_solution(self, problem):
        raise NotImplementedError

    def solve(self, problem):
        """Find problem solution."""
        self._before_solution(problem)
        # pylint: disable=no-member
        if self._options["Deterministic"]:
            prng = numpy.random.RandomState(self._options["Seed"])
        else:
            prng = numpy.random.RandomState()
        technique = self._options["Technique"]
        if technique == "Random search":
            self.__random_search(problem, prng)
        elif technique == "Genetic":
            self.__genetic(problem)
        return self._after_solution(problem)

    def __random_search(self, problem, prng):
        # random generation
        for _ in range(self._options["Budget"]):
            x = numpy.array(
                [prng.uniform(var["lb"], var["ub"]) for var in problem.variables]
            )
            f = problem.evaluate_objectives(x)
            if len(f) != len(problem.objectives):
                raise RuntimeError("Wrong number of objectives evaluated.")
            c = problem.evaluate_constraints(x)
            if len(c) != len(problem.constraints):
                raise RuntimeError("Wrong number of constraints evaluated.")
            self._after_iteration(problem, x, f, c)

    @staticmethod
    def __genetic(problem):
        del problem
        raise RuntimeError("Genetic algorithm is not implemented yet.")


class DSXTestWrite(DSX):
    """DSX with history write."""

    def __init__(self, options, path):
        super(DSXTestWrite, self).__init__(options)
        self.__history = []
        self.__history_path = os.path.join(path, "history.p7data")

    def _before_solution(self, problem):
        pass

    def _after_iteration(self, problem, x, f, c):
        self.__history.append(x.tolist() + f.tolist() + c.tolist() + [True, True])

    def __fill_feasible(self, problem):
        tolerance = self._options["Constraints tolerance"]
        feasible_column = -2

        def feasible(j, con):
            return (
                problem.constraints[j]["lb"] * (1 - tolerance)
                <= con
                <= problem.constraints[j]["ub"] * (1 + tolerance)
            )

        for i, iteration in enumerate(self.__history):
            self.__history[i][feasible_column] = all(
                [
                    feasible(j, con)
                    for j, con in enumerate(iteration[problem.constraints_slice])
                ]
            )

    def __fill_optimal(self, problem):
        feasible_column = -2
        optimal_column = -1

        def dominate(l1, l2):
            return all(val <= l2[j] for j, val in enumerate(l1)) and any(
                val < l2[j] for j, val in enumerate(l1)
            )

        for i, iteration in enumerate(self.__history):
            if iteration[feasible_column]:
                for compare_with in self.__history:
                    if (
                        compare_with[feasible_column]
                        and compare_with[optimal_column]
                        and dominate(
                            compare_with[problem.objectives_slice],
                            iteration[problem.objectives_slice],
                        )
                    ):
                        self.__history[i][optimal_column] = False
                        break

    def _after_solution(self, problem):
        self.__fill_feasible(problem)
        self.__fill_optimal(problem)
        dtype = []
        dtype.extend([(var["name"], numpy.float64) for var in problem.variables])
        dtype.extend([(var["name"], numpy.float64) for var in problem.objectives])
        dtype.extend([(var["name"], numpy.float64) for var in problem.constraints])
        dtype.extend([("feasible", numpy.bool), ("optimal", numpy.bool)])
        with file.File(self.__history_path, "w") as hf:
            hf.write(
                value=numpy.array([tuple(item) for item in self.__history], dtype=dtype)
            )
        return self.__history_path


class DSXTestUpdate(DSX):
    """DSX with history update."""

    def __init__(self, options, path):
        super(DSXTestUpdate, self).__init__(options)
        self.__history_path = os.path.join(path, "history.p7data")

    def _before_solution(self, problem):
        dtype = []
        dtype.extend([(var["name"], "f8") for var in problem.variables])
        dtype.extend([(var["name"], "f8") for var in problem.objectives])
        dtype.extend([(var["name"], "f8") for var in problem.constraints])
        dtype.extend([("status", "i8")])
        with file.File(self.__history_path, "w") as hf:
            hf.write(value=numpy.array([], dtype=dtype))

    def _after_iteration(self, problem, x, f, c):
        tolerance = self._options["Constraints tolerance"]

        def invalid():
            return numpy.isnan(f).any() or numpy.isnan(c).any()

        def feasible():
            return (problem.constraints_lb * (1 - tolerance) <= c).all() and (
                c <= problem.constraints_ub * (1 + tolerance)
            ).all()

        def dominate(l1, l2):
            return (l1 <= l2).all() and (l1 < l2).any()

        def make_row(status):
            return x.tolist() + f.tolist() + c.tolist() + [status]

        with file.File(self.__history_path, "w") as hf:
            table = hf.get().data
            if invalid():
                table.append(make_row(3))
            else:
                if feasible():
                    for i, row in enumerate(table):
                        if row[0] == 0:
                            objectives = list(row.data[problem.objectives_names][0])
                            if dominate(f, objectives):
                                # status column is the last one
                                table[i, -1] = 1
                            elif dominate(objectives, f):
                                table.append(make_row(1))
                                break
                    else:
                        table.append(make_row(0))
                else:
                    table.append(make_row(2))

    def _after_solution(self, problem):
        return self.__history_path

    def __visualize_2d(self, problem):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.grid(True)
        name1 = problem.objectives_names[0]
        name2 = problem.objectives_names[1]
        with file.File(self.__history_path, "r") as f:
            data = f[f.ids()[0]][0].data
            ax.plot(data[name1], data[name2], "yo", label="All points")
            ax.plot(
                data[name1][data["status"] < 2],
                data[name2][data["status"] < 2],
                "bo",
                label="Feasible points",
            )
            ax.plot(
                data[name1][data["status"] == 0],
                data[name2][data["status"] == 0],
                "ro",
                label="Pareto points",
            )
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.legend(loc="best")
            plt.show()


class DSXTestMask(DSX):
    """DSX with masked table as output."""

    def __init__(self, options, path):
        super(DSXTestMask, self).__init__(options)
        self.__history_path = os.path.join(path, "history.p7data")
        self.__dtype = None

    def _before_solution(self, problem):
        self.__dtype = []
        self.__dtype.extend([(var["name"], "f8") for var in problem.variables])
        self.__dtype.extend([(var["name"], "f8") for var in problem.objectives])
        self.__dtype.extend([(var["name"], "f8") for var in problem.constraints])
        self.__dtype.extend([("status", "U10")])
        properties = {
            "@schema": {"@type": "Table"},
            "@columns": {
                "status": {
                    "@schema": {
                        "@type": "Enumeration",
                        "@values": ["feasible", "infeasible"],
                    }
                }
            },
        }
        for var in problem.variables:
            properties["@columns"][var["name"]] = {
                "@schema": {
                    "@type": "Real",
                    "@minimum": var["lb"],
                    "@maximum": var["ub"],
                }
            }
        with file.File(self.__history_path, "w") as hf:
            hf.write(value=numpy.array([], dtype=self.__dtype), properties=properties)

    def _after_iteration(self, problem, x, f, c):
        tolerance = self._options["Constraints tolerance"]

        def feasible():
            return (problem.constraints_lb * (1 - tolerance) <= c).all() and (
                c <= problem.constraints_ub * (1 + tolerance)
            ).all()

        # at first, write objectives
        value_f = [
            tuple(
                x.tolist()
                + f.tolist()
                + [0.0 for _ in problem.constraints]
                + (["feasible"] if feasible() else ["infeasible"])
            )
        ]
        mask_f = [
            tuple(
                [False for _ in problem.variables]
                + [False for _ in problem.objectives]
                + [True for _ in problem.constraints]
                + [False]
            )
        ]
        point_f = numpy.ma.array(value_f, dtype=self.__dtype, mask=mask_f)
        # then, write constraints
        value_c = [
            tuple(
                x.tolist()
                + [0.0 for _ in problem.objectives]
                + c.tolist()
                + (["feasible"] if feasible() else ["infeasible"])
            )
        ]
        mask_c = [
            tuple(
                [False for _ in problem.variables]
                + [True for _ in problem.objectives]
                + [False for _ in problem.constraints]
                + [False]
            )
        ]
        point_c = numpy.ma.array(value_c, dtype=self.__dtype, mask=mask_c)
        with file.File(self.__history_path, "w") as hf:
            table = hf.get().data
            table.append(point_f)
            table.append(point_c)

    def _after_solution(self, problem):
        return self.__history_path


class TestDSX(BaseTestCase):
    """Base test for DSX."""

    def setUp(self):
        self.__dirname = tempfile.mkdtemp()
        self.__problem = Problem(
            [{"name": "x0", "lb": 0.1, "ub": 1}, {"name": "x1", "lb": 0, "ub": 0.5}],
            [{"name": "f1"}, {"name": "f2"}],
            [
                {"name": "c1", "lb": 0.0, "ub": float("Inf")},
                {"name": "c2", "lb": 0.0, "ub": float("Inf")},
            ],
            lambda x: numpy.array([x[0], (1 + x[1]) / x[0]]),
            lambda x: numpy.array([x[1] + 9 * x[0] - 6, -x[1] + 9 * x[0] - 1]),
        )
        self.__options = {
            "Technique": "Random search",
            "Constraints tolerance": 1e-5,
            "Budget": 100,
            "Deterministic": True,
            "Seed": 100,
        }

    def tearDown(self):
        shutil.rmtree(self.__dirname)

    def test_write(self):
        """Write history table."""

        dsx = DSXTestWrite(self.__options, self.__dirname)
        path = dsx.solve(self.__problem)
        history = file.File(path, "r").read()
        print(history)
        self.assertEmptyMask(history)

    def test_update(self):
        """Update history with new run."""

        dsx = DSXTestUpdate(self.__options, self.__dirname)
        path = dsx.solve(self.__problem)
        history = file.File(path, "r").read()
        print(history)
        self.assertEmptyMask(history)

    def test_mask(self):
        """Test masked output."""

        dsx = DSXTestMask(self.__options, self.__dirname)
        path = dsx.solve(self.__problem)
        history = file.File(path, "r").read()
        print(history)

        # copy history by columns for avoiding numpy future warnings
        dtype = [(name, numpy.bool_) for name in self.__problem.names]
        mask = numpy.zeros((len(history)), dtype=dtype)
        dtype = [(name, history.dtype[name]) for name in self.__problem.names]
        data = numpy.ma.masked_array(
            numpy.zeros((len(history)), dtype=dtype), mask=mask
        )
        for name in data.dtype.names:
            data[name] = history[name].copy()

        data = data.view("f8").reshape(history.size, len(history.dtype) - 1)
        self.assertTrue(data[::2, self.__problem.constraints_slice].mask.all())
        self.assertFalse(data[::2, self.__problem.objectives_slice].mask.any())
        self.assertTrue(data[1::2, self.__problem.objectives_slice].mask.all())
        self.assertFalse(data[1::2, self.__problem.constraints_slice].mask.any())
        statuses = numpy.unique(history["status"].data)
        self.assertCountEqual(statuses, ["feasible", "infeasible"])


class TestOptions(BaseTestCase):
    """Test settings stored as p7 value."""

    def setUp(self):
        self.__dirname = tempfile.mkdtemp()
        # @todo check several @schemas,
        # use Seed option (from numpy docs: can be an integer,
        # an array (or other sequence) of integers of any length)
        self.__options = [
            {
                "@name": "Technique",
                "@schema": {
                    "@type": "Enumeration",
                    "@name": "Enumeration",
                    "@nullable": False,
                    "@init": "Random search",
                    "@values": ["Random search", "Genetic"],
                },
            },
            {
                "@name": "Constraints tolerance",
                "@schema": {
                    "@type": "Real",
                    "@name": "Real",
                    "@nullable": False,
                    "@init": 0.0,
                    "@minimum": 0.0,
                    "@exclusive_minimum": True,
                    "@exclusive_maximum": True,
                    "@maximum": 1.0,
                    "@nanable": False,
                },
            },
            {
                "@name": "Budget",
                "@schema": {
                    "@type": "Integer",
                    "@name": "Integer",
                    "@nullable": False,
                    "@minimum": 1,
                    "@init": 1000,
                },
            },
            {
                "@name": "Deterministic",
                "@schema": {
                    "@type": "Boolean",
                    "@name": "Boolean",
                    "@nullable": False,
                    "@init": False,
                },
            },
            {
                "@name": "Seed",
                "@schema": {
                    "@type": "Integer",
                    "@name": "Integer",
                    "@nullable": False,
                    "@init": 0,
                    "@minimum": 0,
                    "@maximum": 4294967295,
                },
            },
        ]

    def tearDown(self):
        shutil.rmtree(self.__dirname)

    def test_read(self):
        """Read options."""

        values = {
            "Technique": "Random search",
            "Constraints tolerance": 1e-5,
            "Budget": 10,
            "Deterministic": True,
            "Seed": 100,
        }
        # write options values and properties to file, all values are valid
        path = os.path.join(self.__dirname, "options.p7data")
        to_write = [
            {"id": uuid.uuid4(), "value": values[option["@name"]], "properties": option}
            for option in self.__options
        ]
        of = file.File(path, "w")
        for args in to_write:
            of.write(**args)
        # read options
        id_to_name = {
            option["id"]: option["properties"]["@name"] for option in to_write
        }
        read_values = {}
        with of:
            for id in of.ids():
                read_values[id_to_name[id]] = of[id]
        # for a start, check that values are equal
        self.assertEqual(read_values, values)
        # @todo check read properties
        # @todo pass invalid arguments and check validation

    def _read_write_convert(self, values):
        path = os.path.join(self.__dirname, "options.p7data")
        to_write = [
            {
                "id": uuid.uuid4(),
                "value": values[option["@name"]],
                "name": option["@name"],
            }
            for option in self.__options
        ]
        of = file.File(path, "w")
        for args in to_write:
            of.write(args["id"], args["value"])
        id_to_name = {option["id"]: option["name"] for option in to_write}
        name_to_schema = {
            option["@name"]: option["@schema"] for option in self.__options
        }
        read_values = {}
        with of:
            for id, name in id_to_name.items():
                read_values[name] = convert(name_to_schema[name], of.read(id))
        return read_values

    def test_convert(self):
        """Convert options."""

        reference_values = {
            "Technique": "Random search",
            "Constraints tolerance": 1e-5,
            "Budget": 10,
            "Deterministic": True,
            "Seed": 100,
        }

        values = {
            "Technique": numpy.array(["Random search"]),
            "Constraints tolerance": numpy.array([[1e-5]]),
            "Budget": 10.0,
            "Deterministic": 1,
            "Seed": [100],
        }
        self.assertEqual(self._read_write_convert(values), reference_values)

        values = {
            "Technique": numpy.array([["Random search"]]),
            "Constraints tolerance": [[1e-5]],
            "Budget": "10",
            "Deterministic": "True",
            "Seed": numpy.array([100]),
        }
        self.assertEqual(self._read_write_convert(values), reference_values)

    def test_properties(self):
        """Write and read options."""

        path = os.path.join(self.__dirname, "test.p7data")
        value = {"mu": 0.5, "sigma": 1}
        properties = {
            "@schema": {
                "@type": "Structure",
                "@init": [{"key": "mu", "value": 0.0}, {"key": "sigma", "value": 1.0}],
                "@name": "Normal",
                "@nullable": False,
                "@schema": {
                    "mu": [
                        {
                            "@type": "Real",
                            "@name": "mu",
                            "@nullable": False,
                            "@init": 0.0,
                        }
                    ],
                    "sigma": [
                        {
                            "@type": "Real",
                            "@name": "sigma",
                            "@nullable": False,
                            "@init": 1.0,
                            "@minimum": 0.0,
                            "@exclusive_minimum": True,
                        }
                    ],
                },
            }
        }
        file.File(path, "w").write(value=value, properties=properties)
        self.assertEqual(value, file.File(path, "w").read())
        with file.File(path, "r") as f:
            self.assertEqual(properties, f.get().properties)
