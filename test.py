import unittest
from convert import make_structs


class TestParse(unittest.TestCase):
    def test_parse_simple(self):
        parser_input = {"weight": None, "bias": None}
        expected = {"module": {"weight": type(None), "bias": type(None)}}

        self.assertEqual(make_structs(parser_input), expected)

    def test_parse_struct(self):
        parser_input = {"linear": {"weight": None, "bias": None}}
        expected = {"module": {"linear": "struct Linear"}, "Linear": {"weight": type(None), "bias": type(None)}}

        self.assertEqual(make_structs(parser_input), expected)

    def test_parse_nested_struct(self):
        parser_input = {"block": {"linear": {"weight": None, "bias": None}}}
        expected = {"module": {"block": "struct Block"}, "Block": {"linear": "struct Linear"}, "Linear": {"weight": type(None), "bias": type(None)}}

        self.assertEqual(make_structs(parser_input), expected)

    def test_parse_array(self):
        parser_input = {"conv0": {'0': {"weight": None, "bias": None}}}
        expected = {"module": {"conv0": "struct Conv0[1]"}, "Conv0": {"weight": type(None), "bias": type(None)}}

        self.assertEqual(make_structs(parser_input), expected)


if __name__ == '__main__':
    unittest.main()
