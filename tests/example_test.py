import unittest

class ExampleTest(unittest.TestCase):
    def test_always_pass(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()