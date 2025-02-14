import unittest
import sys
sys.path.append("/path/to/directory")

class TestGaussianMixture:
    super(unittest.TestCase)
    def __init__(self,gm_module):
        self.gm_module = gm_module
        self.gm1 = gm_module([[0]],[[1]],[1])
    def standard_normal_1d(self):
        self.assertTrue(self.gm1.evaluate([[1]]))
        

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

unittest.main()